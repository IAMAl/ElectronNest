##################################################################
##
##	ElectronNest_CP
##	Copyright (C) 2024  Shigeyuki TAKANO
##
##  GNU AFFERO GENERAL PUBLIC LICENSE
##	version 3.0
##
##################################################################
from typing import TypedDict, List, Dict, Tuple, Optional, Set, Union, Any
import os
import sys
import logging


# Configure logging
def setup_logger():
	# Create logger
	logger = logging.getLogger('DataPathGenerator')
	logger.setLevel(logging.INFO)

	# Create console handler and set level to debug
	ch = logging.StreamHandler(sys.stdout)
	ch.setLevel(logging.INFO)

	# Create formatter
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

	# Add formatter to ch
	ch.setFormatter(formatter)

	# Add ch to logger
	logger.addHandler(ch)

	return logger

# Initialize logger
logger = setup_logger()

class DataPathGenerator:
	def __init__(self, array_patterns, compute_paths, r_file_path, r_name):
		self.array_patterns = array_patterns['array_patterns']
		print(f"init compute_paths:{compute_paths}")
		self.compute_paths = compute_paths
		self.loop_levels = array_patterns['loop_levels']
		self.loops = array_patterns['loops']
		self.r_path = r_file_path
		self.r_name = r_name
		self.array_dims = self._get_array_dimensions()  # ここで呼び出し

	def _get_array_dimensions(self) -> Dict[str, List[int]]:
		"""
		配列の次元情報を取得
		Returns:
			Dict[str, List[int]]: {array_name: [dim1, dim2, ...]}
		"""
		try:
			array_dims = {}
			for array_name, pattern in self.array_patterns.items():
				dims = []
				if 'array_info' in pattern:
					array_info = pattern['array_info']
					# loop_accessから次元サイズを収集
					for access in array_info.get('loop_access', {}).values():
						if 'array_size' in access:
							dims.append(access['array_size'])

				if dims:  # 次元情報が得られた場合のみ追加
					array_dims[array_name] = dims

			return array_dims

		except Exception as e:
			logger.error(f"Error getting array dimensions: {e}")
			return {}

	def _get_loop_bound(self, loop_level: str) -> int:
		"""
		ループレベルに対応する境界値を取得
		"""
		try:
			if loop_level in self.loop_levels:
				for array_name, pattern in self.array_patterns.items():
					loop_access = pattern.get('array_info', {}).get('loop_access', {})
					if loop_level in loop_access:
						return loop_access[loop_level].get('array_size', 32)
			return 32  # デフォルト値

		except Exception as e:
			logger.error(f"Error getting loop bound: {e}")
			return 32

	def ComputeDataPath(self) -> Dict:
		"""
		計算パスを解析し、データパスLLVMコードを生成する。
		複数の基本ブロックにまたがるデータパスの処理に対応。
		各基本ブロック内のロード・ストア命令のみを対象とし、
		計算本体の書き込み先レジスタ番号をストアする命令で、
		かつ計算本体の書き込み先レジスタ番号をストアソースにしている命令だけ生成する。
		命令の順序は依存関係とソースコードの構造に基づいて決定する。
		"""
		result = {'code': [], 'structure': {'entry': None, 'exit': None}}
		try:
			# 計算操作を取得
			operations = []
			if isinstance(self.compute_paths, dict) and 'compute_info' in self.compute_paths:
				operations = self.compute_paths['compute_info'].get('operations', [])

			# メモリ操作情報も取得
			memory_ops = {}
			if isinstance(self.compute_paths, dict) and 'compute_info' in self.compute_paths:
				memory_ops = self.compute_paths['compute_info'].get('memory_ops', {})

			# 基本ブロックごとの操作をグループ化
			block_operations = {}  # block_id -> operations
			block_loads = {}       # block_id -> loads
			block_stores = {}      # block_id -> stores

			# 操作のブロック情報を収集
			for op in operations:
				block_id = self._get_block_for_operation(op)
				if block_id:
					if block_id not in block_operations:
						block_operations[block_id] = []
					block_operations[block_id].append(op)

			# ロード操作のブロック情報を収集
			for load in memory_ops.get('loads', []):
				block_id = self._get_block_for_memory_op(load)
				if block_id:
					if block_id not in block_loads:
						block_loads[block_id] = []
					block_loads[block_id].append(load)

			# ストア操作のブロック情報を収集
			for store in memory_ops.get('stores', []):
				block_id = self._get_block_for_memory_op(store)
				if block_id:
					if block_id not in block_stores:
						block_stores[block_id] = []
					block_stores[block_id].append(store)

			# 処理対象のブロックを取得
			all_blocks = set(block_operations.keys()) | set(block_loads.keys()) | set(block_stores.keys())

			# 基本ブロックの順序を特定（ループ構造から抽出）
			block_order = self._determine_block_execution_order(all_blocks)

			# 基本ブロックごとに命令を生成
			for block_id in block_order:
				# このブロックに必要な操作を収集
				ops = block_operations.get(block_id, [])
				loads = block_loads.get(block_id, [])
				stores = block_stores.get(block_id, [])

				# 基本ブロック開始コメント
				#result['code'].append(f"; Basic Block {block_id}")

				# 有効な計算命令を収集
				computation_ops = []
				for op in ops:
					opcode = op.get('op', '').split('_')[0]
					if opcode in {'add', 'sub', 'mul', 'div', 'and', 'or', 'xor', 'shl', 'ashr', 'lshr'} and 'output' in op and 'inputs' in op:
						if len(op.get('inputs')) > 1:
							computation_ops.append(op)

				# 各レジスタの定義元となる命令をマップ
				reg_defined_by = {}
				for op in computation_ops:
					output_reg = op.get('output')
					if output_reg:
						reg_defined_by[output_reg] = op

				# ロード操作の処理
				load_ptr_map = {}  # ロードレジスタ -> ポインタレジスタのマッピング
				for load in loads:
					if 'reg' in load and 'source_reg' in load:
						load_ptr_map[load.get('reg')] = load.get('source_reg')

				# このブロックで必要なロードを特定
				required_loads = set()
				for op in computation_ops:
					for input_reg in op.get('inputs', []):
						if input_reg.startswith('%') and input_reg not in reg_defined_by:
							required_loads.add(input_reg)

				# ロード命令の順序を決定する
				ordered_loads = self._determine_load_order(required_loads, load_ptr_map, computation_ops)

				# ロード命令を生成
				for load_reg, ptr_reg in ordered_loads:
					# 配列名を取得
					array_name = self._get_array_for_reg(load_reg)

					if array_name:
						# 配列名とオリジナルのポインタレジスタ番号を組み合わせる
						ptr_name = f"%{array_name}_{ptr_reg[1:]}"
					else:
						# 配列名がない場合はポインタレジスタをそのまま使用
						ptr_name = ptr_reg

					result['code'].append(f"{load_reg} = load i32, i32* {ptr_name}, align 4")

				# 計算命令の依存関係を解決
				dependency_count = {}
				for op in computation_ops:
					dependency_count[id(op)] = 0
					for input_reg in op.get('inputs', []):
						if input_reg in reg_defined_by:
							dependency_count[id(op)] += 1

				# 依存関係のない命令から処理開始
				ready = [op for op in computation_ops if dependency_count[id(op)] == 0]
				ordered_ops = []

				while ready:
					current = ready.pop(0)
					ordered_ops.append(current)

					output_reg = current.get('output')
					for op in computation_ops:
						if op not in ordered_ops and op not in ready:
							if output_reg in op.get('inputs', []):
								dependency_count[id(op)] -= 1
								if dependency_count[id(op)] == 0:
									ready.append(op)

				# 依存関係が解決できなかった命令も追加
				if len(ordered_ops) != len(computation_ops):
					for op in computation_ops:
						if op not in ordered_ops:
							ordered_ops.append(op)

				# 計算命令を依存関係順に生成
				for op in ordered_ops:
					opcode = op.get('op', '').split('_')[0]
					output = op.get('output')
					inputs = op.get('inputs', [])

					if len(inputs) >= 1:
						if len(inputs) >= 2:
							result['code'].append(f"{output} = {opcode} i32 {inputs[0]}, {inputs[1]}")
						else:
							result['code'].append(f"{output} = {opcode} i32 {inputs[0]}, 0")

				# このブロックでの計算結果を特定
				final_computation_result = None
				if ordered_ops:
					final_computation_result = ordered_ops[-1].get('output')

				# ストア命令の処理
				if final_computation_result and stores:
					# 条件に合うストア命令のみを選択
					valid_stores = []
					for store in stores:
						value_reg = store.get('value_reg')

						# 明示的に計算結果を値として使用するストア命令
						if value_reg == final_computation_result:
							valid_stores.append(store)

					# 該当するストア命令がない場合は、未指定の値を持つストア命令を探す
					if not valid_stores:
						for store in stores:
							value_reg = store.get('value_reg')
							# 値が未指定または空の場合
							if not value_reg or value_reg == 'None' or value_reg == '':
								# 計算結果を値として使用する
								store['value_reg'] = final_computation_result
								valid_stores.append(store)
								break

					# 選択されたストア命令を生成
					for store in valid_stores:
						target_reg = store.get('target_reg')
						value_reg = store.get('value_reg')
						array_name = store.get('array', '')

						if array_name:
							ptr_name = f"%{array_name}_{target_reg[1:]}"
						else:
							ptr_name = target_reg

						result['code'].append(f"store i32 {value_reg}, i32* {ptr_name}, align 4")

				# 基本ブロック終了コメント
				result['code'].append("")  # 空行

			return result

		except Exception as e:
			import traceback
			print(f"ComputeDataPath error: {str(e)}")
			traceback.print_exc()
			return result

	def _determine_load_order(self, required_loads, load_ptr_map, computation_ops):
		"""
		ロード命令の実行順序を決定する。依存関係と計算パターンに基づいて順序付けを行う。

		Args:
			required_loads: 必要なロードレジスタのセット
			load_ptr_map: ロードレジスタからポインタレジスタへのマッピング
			computation_ops: 計算命令のリスト

		Returns:
			List[Tuple[str, str]]: 順序付けされた(ロードレジスタ, ポインタレジスタ)のリスト
		"""
		try:
			# 計算命令での使用順序を考慮する
			reg_usage_order = []

			# 計算パターンを分析
			if len(computation_ops) >= 2:
				# オペコードを取得
				opcodes = [op.get('op', '').split('_')[0] for op in computation_ops]

				# 計算命令の依存関係を解析
				dependency_graph = {}
				for op in computation_ops:
					output = op.get('output')
					dependency_graph[output] = set()

					# この命令の出力に依存する命令を特定
					for dep_op in computation_ops:
						if dep_op is not op:
							if output in dep_op.get('inputs', []):
								dependency_graph[output].add(dep_op.get('output'))

				# 計算グラフのリーフノード（他の命令に入力されない出力レジスタ）を特定
				leaf_nodes = set()
				for output, deps in dependency_graph.items():
					if not deps:
						leaf_nodes.add(output)

				# リーフノードから逆方向に依存関係を辿り、入力レジスタの使用順序を特定
				visited = set()

				def visit_dependencies(node):
					if node in visited:
						return
					visited.add(node)

					# この命令を探す
					for op in computation_ops:
						if op.get('output') == node:
							# 入力レジスタを使用順序に追加
							for input_reg in reversed(op.get('inputs', [])):
								if input_reg in required_loads and input_reg not in reg_usage_order:
									reg_usage_order.append(input_reg)

							# 依存元を再帰的に訪問
							for dep_op in computation_ops:
								if dep_op is not op and node in dep_op.get('inputs', []):
									visit_dependencies(dep_op.get('output'))

				# リーフノードから逆方向に辿る
				for leaf in leaf_nodes:
					visit_dependencies(leaf)

				# リストを反転して正しい順序にする
				reg_usage_order.reverse()

			# 使用順序が見つからない場合、単純に命令の入力順で追加
			if not reg_usage_order:
				for op in computation_ops:
					for input_reg in op.get('inputs', []):
						if input_reg in required_loads and input_reg not in reg_usage_order:
							reg_usage_order.append(input_reg)

			# 最終的なロード順序を構築
			ordered_loads = []
			for reg in reg_usage_order:
				if reg in load_ptr_map:
					ordered_loads.append((reg, load_ptr_map[reg]))

			# まだ処理されていないロードを追加
			for reg in required_loads:
				if reg in load_ptr_map and reg not in reg_usage_order:
					ordered_loads.append((reg, load_ptr_map[reg]))

			return ordered_loads

		except Exception as e:
			print(f"Error determining load order: {e}")
			# エラー時は単純にロードを順番に並べる
			return [(reg, load_ptr_map[reg]) for reg in required_loads if reg in load_ptr_map]

	def _get_block_for_operation(self, operation: Dict) -> Optional[str]:
		"""
		計算操作の所属する基本ブロックIDを取得

		Args:
			operation: 操作情報

		Returns:
			基本ブロックIDまたはNone
		"""
		try:
			# 操作からブロック情報を直接取得
			if 'block_id' in operation:
				return operation['block_id']

			# path_idからブロックIDを抽出 (例: path_22_add_1 -> 22)
			if 'path_id' in operation:
				path_id = operation['path_id']
				parts = path_id.split('_')
				if len(parts) > 1 and parts[0] == 'path' and parts[1].isdigit():
					return parts[1]

			# compute_pathsからの検索
			if isinstance(self.compute_paths, dict) and 'compute_paths' in self.compute_paths:
				for path in self.compute_paths['compute_paths']:
					if 'path_id' in path and path['path_id'].startswith('path_'):
						# ブロックIDを抽出
						block_id = path['path_id'].split('_')[1]
						if block_id.isdigit():
							# このパスに含まれる操作と一致するか確認
							for comp in path.get('computation', {}).get('sequence', []):
								if (comp.get('opcode') == operation.get('op') and
									comp.get('output_reg') == operation.get('output') and
									comp.get('input_regs') == operation.get('inputs')):
									return block_id

			# ブロック情報が見つからない場合
			return self._get_default_block_id()

		except Exception as e:
			print(f"Error getting block for operation: {e}")
			return self._get_default_block_id()

	def _get_block_for_memory_op(self, mem_op: Dict) -> Optional[str]:
		"""
		メモリ操作（ロード/ストア）の所属する基本ブロックIDを取得

		Args:
			mem_op: メモリ操作情報

		Returns:
			基本ブロックIDまたはNone
		"""
		try:
			# 操作からブロック情報を直接取得
			if 'block_id' in mem_op:
				return mem_op['block_id']

			# compute_pathsからの検索
			if isinstance(self.compute_paths, dict) and 'compute_paths' in self.compute_paths:
				for path in self.compute_paths['compute_paths']:
					if 'path_id' in path and path['path_id'].startswith('path_'):
						# ブロックIDを抽出
						block_id = path['path_id'].split('_')[1]
						if block_id.isdigit():
							# このパスに含まれるメモリ操作と一致するか確認
							if 'reg' in mem_op and path.get('type') == 'load':
								for load in path.get('inputs', {}).get('loads', []):
									if load.get('reg') == mem_op.get('reg'):
										return block_id

							if 'target_reg' in mem_op and path.get('output', {}).get('type') == 'memory':
								if path['output'].get('target_reg') == mem_op.get('target_reg'):
									return block_id

			# ブロック情報が見つからない場合はデフォルトを使用
			return self._get_default_block_id()

		except Exception as e:
			print(f"Error getting block for memory op: {e}")
			return self._get_default_block_id()

	def _get_default_block_id(self):
		"""
		デフォルトのブロックIDを取得する。
		入力データから適切なブロックIDを推測する。
		"""
		# 既に計算済みの場合はそれを返す
		if hasattr(self, 'default_block_id'):
			return self.default_block_id

		# 1. compute_pathsから最初のブロックIDを取得
		if isinstance(self.compute_paths, dict) and 'compute_paths' in self.compute_paths:
			for path in self.compute_paths['compute_paths']:
				if 'path_id' in path and path['path_id'].startswith('path_'):
					parts = path['path_id'].split('_')
					if len(parts) > 1 and parts[1].isdigit():
						self.default_block_id = parts[1]
						return self.default_block_id

		# 2. ループ情報から最初のブロックを取得
		if hasattr(self, 'loops') and self.loops:
			for loop_info in self.loops:
				if 'nodes' in loop_info and loop_info['nodes']:
					self.default_block_id = loop_info['nodes'][0]
					return self.default_block_id

		# 3. すべて失敗した場合は最もシンプルなデフォルト値
		self.default_block_id = "1"
		return self.default_block_id

	def _determine_block_execution_order(self, blocks: Set[str]) -> List[str]:
		"""
		基本ブロックの実行順序を決定

		Args:
			blocks: 処理対象のブロックIDのセット

		Returns:
			ブロックIDのリスト（実行順序）
		"""
		try:
			# ブロックの順序をソート（数値として）
			block_order = sorted(blocks, key=lambda x: int(x) if x.isdigit() else float('inf'))

			# ループ構造を考慮した順序付け
			if hasattr(self, 'loops') and self.loops:
				# ループ構造から順序情報を抽出
				loop_order = []
				for loop_info in self.loops:
					if 'nodes' in loop_info:
						# このループに含まれるブロックを順序維持
						for node in loop_info['nodes']:
							if node in blocks and node not in loop_order:
								loop_order.append(node)

				# ループ構造から得られた順序を優先
				if loop_order:
					return loop_order

			return block_order

		except Exception as e:
			print(f"Error determining block execution order: {e}")
			return sorted(list(blocks))  # エラー時は単純にソート

	def _get_array_for_reg(self, reg: str) -> Optional[str]:
		"""
		レジスタに関連付けられた配列名を取得する

		Args:
			reg: レジスタ名

		Returns:
			配列名またはNone
		"""
		try:
			# 配列アクセス情報から検索
			for array_name, pattern in self.array_patterns.items():
				if 'array_info' not in pattern:
					continue

				for loop_level, access_info in pattern['array_info'].get('loop_access', {}).items():
					# レジスタを確認
					registers = access_info.get('registers', {})
					for reg_type in ['gep', 'load', 'store']:
						if reg in registers.get(reg_type, []):
							return array_name

			# 計算パスからも検索
			if isinstance(self.compute_paths, dict) and 'compute_paths' in self.compute_paths:
				for path in self.compute_paths['compute_paths']:
					# ロード情報を確認
					for load in path.get('inputs', {}).get('loads', []):
						if load.get('reg') == reg and 'array' in load:
							return load['array']

					# ストア情報を確認
					if path.get('output', {}).get('type') == 'memory':
						if path['output'].get('target_reg') == reg and 'array' in path['output']:
							return path['output']['array']

			return None
		except Exception as e:
			print(f"Error in _get_array_for_reg for {reg}: {e}")
			return None



def prepare_data_for_generator(analysis_result):
	"""
	Analyzerから得られた結果をDataPathGeneratorに適した形式に変換する

	Args:
		analysis_result (Dict): analyzer.analyze()の結果

	Returns:
		Dict: DataPathGeneratorが期待する形式の辞書
	"""
	# array_patternsの基本構造を初期化
	array_patterns = {
		'array_patterns': {},
		'loop_levels': analysis_result['loop_levels'],
		'loops': []
	}

	# 配列情報の変換
	for array_name, info in analysis_result['array_info'].items():
		array_patterns['array_patterns'][array_name] = {
			'array_info': {
				'dimensions': info['dimensions'],
				'loop_access': {}
			}
		}

		# ループアクセス情報の処理
		for loop_level, access_info in info['loop_access'].items():
			# 次元情報の取得 (該当ループレベルで使用される次元を特定)
			dimension_index = None
			# array_dim_to_loopからこの配列の、このループレベルでアクセスされる次元を探す
			for dim, level in analysis_result.get('array_dim_to_loop', {}).get(array_name, {}).items():
				if level == loop_level:
					dimension_index = int(dim)
					break

			# 配列サイズの決定
			array_size = 32  # デフォルト値
			if info['dimensions'] and dimension_index is not None and dimension_index < len(info['dimensions']):
				array_size = info['dimensions'][dimension_index]

			# ループアクセス情報の構築
			array_patterns['array_patterns'][array_name]['array_info']['loop_access'][loop_level] = {
				'array_size': array_size,
				'registers': {
					'gep': access_info.get('gep_regs', []),
					'load': access_info.get('load_regs', []),
					'store': access_info.get('store_regs', [])
				}
			}

	# ループ構造情報の変換
	for level, loop_info in analysis_result['loop_levels'].items():
		# LoopInfoオブジェクトからデータを抽出
		loop_data = {
			'level': level,
			'nodes': loop_info.nodes,
			'header': loop_info.header,
			'exit': loop_info.exit,
			'parent': loop_info.parent,
			'children': loop_info.children
		}
		array_patterns['loops'].append(loop_data)

	return array_patterns

def prepare_compute_paths(analysis_result):
	"""
	Analyzerから得られた結果から計算パス情報を抽出する

	Args:
		analysis_result (Dict): analyzer.analyze()の結果

	Returns:
		Dict: ComputeDataPathメソッドが期待する形式の辞書
	"""
	# 結果の初期化
	compute_paths = {'compute_paths': [], 'compute_info': {}}
	
	# analysis_resultから直接compute_infoを取得
	if 'compute_info' in analysis_result:
		compute_paths['compute_info'] = analysis_result['compute_info']
		print(f"DEBUG: Copied compute_info with {len(analysis_result['compute_info'].get('operations', []))} operations")
	
	# 計算操作のみを処理
	operations = analysis_result['compute_info'].get('operations', []) if 'compute_info' in analysis_result else []
	print(f"DEBUG: Found {len(operations)} operations")
	
	for op in operations:
		opcode = op.get('op', '').split('_')[0]  # 基本的なオペコード部分を取得
		output = op.get('output', '')
		inputs = op.get('inputs', [])
		
		# 計算命令（算術・論理演算）を処理
		if opcode in {'add', 'sub', 'mul', 'div', 'and', 'or', 'xor', 'shl', 'ashr', 'lshr'}:
			# 命令タイプを決定
			path_type = 'computation'
			if opcode == 'mul':
				path_type = 'multiply'
			elif opcode == 'add':
				path_type = 'add_chain'
			
			# 計算パスを追加
			compute_path = {
				'path_id': f"path_{path_type}_{output}",
				'type': path_type,
				'inputs': {'loads': [], 'leafs': []},
				'computation': {
					'sequence': [{
						'opcode': op.get('op', ''),
						'output_reg': output,
						'input_regs': inputs
					}]
				},
				'output': {'type': 'register', 'value_reg': output},
				'loop_context': {'level': None}
			}
			
			compute_paths['compute_paths'].append(compute_path)
			print(f"DEBUG: Added {path_type} path for {output}: {inputs}")
		
		# 比較命令の処理
		elif opcode == 'icmp':
			compute_path = {
				'path_id': f"path_comparison_{output}",
				'type': 'comparison',
				'inputs': {'loads': [], 'leafs': []},
				'computation': {
					'sequence': [{
						'opcode': op.get('op', ''),
						'output_reg': output,
						'input_regs': inputs
					}]
				},
				'output': {'type': 'register', 'value_reg': output},
				'loop_context': {'level': None}
			}
			
			compute_paths['compute_paths'].append(compute_path)
			print(f"DEBUG: Added comparison path for {output}: {inputs}")
	
	# メモリ操作も処理
	memory_ops = analysis_result['compute_info'].get('memory_ops', {})
	
	# ロード操作
	loads = memory_ops.get('loads', [])
	print(f"DEBUG: Found {len(loads)} load operations")
	for i, load in enumerate(loads):
		if 'reg' in load:
			compute_path = {
				'path_id': f"path_load_{load['reg']}",
				'type': 'load',
				'inputs': {'loads': [load], 'leafs': []},
				'computation': {'sequence': []},
				'output': {'type': 'register', 'value_reg': load.get('reg', '')},
				'loop_context': {'level': None}
			}
			
			compute_paths['compute_paths'].append(compute_path)
			print(f"DEBUG: Added load path for {load['reg']}")
	
	# ストア操作
	stores = memory_ops.get('stores', [])
	print(f"DEBUG: Found {len(stores)} store operations")
	for i, store in enumerate(stores):
		if store and 'target_reg' in store:
			target_reg = store.get('target_reg', '')
			compute_path = {
				'path_id': f"path_store_{target_reg}",
				'type': 'store',
				'inputs': {'loads': [], 'leafs': []},
				'computation': {'sequence': []},
				'output': store,
				'loop_context': {'level': None}
			}
			
			compute_paths['compute_paths'].append(compute_path)
			print(f"DEBUG: Added store path for {target_reg}")
	
	print(f"DEBUG: Generated {len(compute_paths['compute_paths'])} compute paths in total")
	return compute_paths


class Gen_DataPath:
    
	def generate_datapath(analyzer_result, r_file_path, r_name):
		"""
		解析結果からデータパスを生成する

		Args:
			analyzer_result (Dict): analyzer.analyze()の結果
			r_file_path (str): 結果ファイルのパス
			r_name (str): 結果ファイルの名前

		Returns:
			Dict: DataPathGeneratorの結果
		"""
		# 入力データの準備
		array_patterns = prepare_data_for_generator(analyzer_result)
		compute_paths = prepare_compute_paths(analyzer_result)
		print(f"generate_datapath compute_paths:{compute_paths}")

		# DataPathGeneratorのインスタンス化
		generator = DataPathGenerator(array_patterns, compute_paths, r_file_path, r_name)

		# データパスの生成
		return generator