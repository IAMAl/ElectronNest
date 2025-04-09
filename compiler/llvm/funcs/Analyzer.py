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
from dataclasses import dataclass, field
import os
import utils.AMUtils as AMUtils
import re
from typing import List, Optional
import functools


@dataclass
class RegisterInfo:
	regs: List[str]			# Register's name
	array_dim: int			# Array Dimension

@dataclass
class PointerInfo:
	block_id: str			# Basic Block ID
	gep_node_id: List[str]	# getelementptr Node-ID
	array_name: str			# Array Name
	index_regs: RegisterInfo

@dataclass
class MemoryOp:
	reg_addr: str			# Register index used for Address
	reg_val: str			# REgister index used for Value

@dataclass
class LoopInfo:
	nodes: List[str]		# Loop CFG Nodes
	header: str				# Loop Header
	exit: str				# Exit Node
	parent: str				# Parent Node-ID
	children: List[str]		# Child Node-ID
	array_dims: Dict[str, int]	# {Array Name: Access Dim}

@dataclass
class DimensionAccess:
	dimension: int          # Access Dim
	loop_level: str         # Loop Level
	array_size: int         # Dim Size
	path_info: Dict[str, Any] = field(default_factory=dict)	# store-to-leafパス情報
															# {
															#   'path_nodes': List[str],	# パス上のノードID
															#   'instructions': List[Dict]	# 各ノードの命令情報
															# }

@dataclass
class ArrayDimInfo:
	array_name: str						# Array Name
	dim_accesses: List[DimensionAccess]	# Dim Info

class IndexExpression:
	base: str					# Base address of the array
	path: List[int]				# List of *node IDs* that compute the index
	source_variables: List[str]	# Source variable for this index expression.

@dataclass
class RegisterFlow:
	reg: str
	opcode: str
	operands: List[str]
	output: str
	extra_info: Dict = field(default_factory=dict)

@dataclass
class CompFlowInfo:
	reg_flows: List[RegisterFlow]
	loop_level: Optional[int] = None
	block_id: Optional[str] = None

	def add_flow(self, reg_flow: RegisterFlow):
		self.reg_flows.append(reg_flow)

class ComputeDataPath:
	def __init__(self, analyzer):
		self.analyzer = analyzer
		self.compute_paths = []
		self.store_load_deps = analyzer.store_load_deps
		self.loop_info = analyzer.loop_levels
		self.all_nodes = analyzer.all_nodes
		self.RegisterFlow = RegisterFlow

	def analyze_compute_paths(self) -> Dict:
		"""計算パスの分析（汎用）"""
		result = {
			'compute_paths': [],
			'path_dependencies': []
		}

		try:
			#print(f"\nDebug: Starting analyze_compute_paths")
			#print(f"Debug: Number of all_nodes: {len(self.analyzer.all_nodes)}")
			#print(f"Debug: Block IDs: {list(self.analyzer.all_nodes.keys())}")

			# 1. ブロックごとの計算パス特定
			for block_id in self.analyzer.all_nodes:
				#print(f"\nDebug: Processing block {block_id}")
				paths = self._identify_block_compute_paths(block_id)
				#print(f"Debug: Found {len(paths)} paths in block {block_id}")
				if paths:
					#print(f"Debug: Paths from block {block_id}:")
					#for path in paths:
					#	#print(f"Debug: - Path ID: {path.get('path_id')}")
					#	#print(f"Debug:   Type: {path.get('type')}")
					#	#print(f"Debug:   Computation sequence: {path.get('computation', {}).get('sequence', [])}")
					result['compute_paths'].extend(paths)

			#print(f"\nDebug: Total paths collected: {len(result['compute_paths'])}")

			# 2. パス間の依存関係分析
			if result['compute_paths']:
				#print(f"Debug: Analyzing path dependencies")
				result['path_dependencies'] = self._analyze_path_dependencies()
				#print(f"Debug: Found {len(result['path_dependencies'])} dependencies")

			#print(f"Debug: Final result structure: {list(result.keys())}")
			#print(f"Debug: Number of compute paths in result: {len(result['compute_paths'])}")

			return result

		except Exception as e:
			print(f"Error in analyze_compute_paths: {e}")
			#print(f"Debug: Exception traceback:", e.__traceback__)
			return result

	def _identify_block_compute_paths(self, block_id: str) -> List[Dict]:
		"""
		基本ブロック内の計算パスを特定する

		Args:
			block_id: 基本ブロックID
		Returns:
			List[Dict]: 特定された計算パスのリスト
		"""
		paths = []
		try:
			nodes = self._read_node_list(block_id)
			if not nodes:
				return paths

			# 配列アクセス情報を追跡
			array_access_info = {}
			load_info = {}  # レジスタ → ロード情報のマッピング
			store_info = {}  # レジスタ → ストア情報のマッピング
			register_to_array = {}  # レジスタ → 配列名のマッピング

			# ステップ1: すべてのGEP、load、store命令を収集する
			for node_idx, node in enumerate(nodes):
				node_info = node[0].split()
				if len(node_info) < 2:
					continue

				# GEP命令の処理 (配列アクセス)
				if 'getelementptr' in node_info[1]:
					dest_reg = node_info[2] if len(node_info) > 2 else None
					array_name = self._get_array_from_reg(node_info[1])
					if array_name and dest_reg:
						array_access_info[dest_reg] = {
							'array': array_name,
							'type': 'gep',
							'reg': dest_reg
						}
						register_to_array[dest_reg] = array_name

				# Load命令の処理
				elif 'load' in node_info[1]:
					if len(node_info) >= 4:
						result_reg = node_info[2]  # ロード結果
						addr_reg = node_info[3]    # アドレス

						# アドレスが配列アクセスから来ている場合
						if addr_reg in array_access_info:
							array_info = array_access_info[addr_reg]
							array_name = array_info.get('array')
							load_info[result_reg] = {
								'array': array_name,
								'reg': result_reg,
								'source_reg': addr_reg
							}
							# 結果レジスタも配列値としてマーク
							array_access_info[result_reg] = {
								'array': array_name,
								'type': 'load',
								'reg': result_reg
							}
							register_to_array[result_reg] = array_name

				# Store命令の処理
				elif 'store' in node_info[1]:
					if len(node_info) >= 4:
						value_reg = node_info[2]  # 格納する値
						addr_reg = node_info[3]   # 格納先アドレス

						# アドレスが配列アクセスから来ている場合
						if addr_reg in array_access_info:
							array_info = array_access_info[addr_reg]
							array_name = array_info.get('array')
							store_info[addr_reg] = {
								'array': array_name,
								'value_reg': value_reg,
								'target_reg': addr_reg
							}

			# ステップ2: 計算パスを構築する
			# 各命令タイプ（load/store/計算）ごとに別々のパスを作成
			# 後で関連付けを行う

			# Load命令の処理
			for node_idx, node in enumerate(nodes):
				node_info = node[0].split()
				if len(node_info) < 2:
					continue

				if 'load' in node_info[1]:
					result_reg = node_info[2]  # ロード結果
					addr_reg = node_info[3]    # アドレス

					# 配列アクセスの場合のみパスを生成
					if addr_reg in array_access_info:
						array_name = array_access_info[addr_reg].get('array')

						load_path = {
							'path_id': f"path_{block_id}_load_{node_idx}",
							'type': 'load',
							'inputs': {
								'loads': [{
									'array': array_name,
									'reg': result_reg,
									'source_reg': addr_reg
								}],
								'leafs': []
							},
							'computation': {
								'sequence': [],
								'flow_info': CompFlowInfo(
									reg_flows=[],
									block_id=block_id
								)
							},
							'output': {
								'type': 'register',
								'value_reg': result_reg
							},
							'loop_context': {
								'level': None,
								'is_reduction': False,
								'is_loop_carried': False
							}
						}

						# ループレベルの設定
						for level, info in self.analyzer.loop_levels.items():
							if block_id in info.nodes:
								load_path['loop_context']['level'] = level
								load_path['computation']['flow_info'].loop_level = int(level)
								break

						paths.append(load_path)

			# Store命令の処理
			for node_idx, node in enumerate(nodes):
				node_info = node[0].split()
				if len(node_info) < 2:
					continue

				if 'store' in node_info[1]:
					value_reg = node_info[2]  # 格納する値
					addr_reg = node_info[3]   # 格納先アドレス

					# 配列アクセスの場合のみパスを生成
					if addr_reg in array_access_info:
						array_name = array_access_info[addr_reg].get('array')

						store_path = {
							'path_id': f"path_{block_id}_store_{node_idx}",
							'type': 'store',
							'inputs': {
								'loads': [],
								'leafs': []
							},
							'computation': {
								'sequence': [],
								'flow_info': CompFlowInfo(
									reg_flows=[],
									block_id=block_id
								)
							},
							'output': {
								'type': 'memory',
								'target_reg': addr_reg,
								'value_reg': value_reg,
								'array': array_name
							},
							'loop_context': {
								'level': None,
								'is_reduction': False,
								'is_loop_carried': False
							}
						}

						# ループレベルの設定
						for level, info in self.analyzer.loop_levels.items():
							if block_id in info.nodes:
								store_path['loop_context']['level'] = level
								store_path['computation']['flow_info'].loop_level = int(level)
								break

						paths.append(store_path)

			# 計算命令の処理
			current_path = None

			for node_idx, node in enumerate(nodes):
				node_info = node[0].split()
				if len(node_info) < 2:
					continue

				operation = node_info[1].split('_')[0]

				# 計算命令（add/mul/sub/icmp）の場合
				if operation in {'add', 'mul', 'sub', 'icmp'}:
					# 新しい計算パスの開始またはリセット
					if current_path is None or operation != current_path['type']:
						# 前のパスが完成していれば追加
						if current_path is not None:
							paths.append(current_path)

						# 新しいパスを初期化
						current_path = {
							'path_id': f"path_{block_id}_{operation}_{node_idx}",
							'type': 'unknown',
							'inputs': {
								'loads': [],
								'leafs': []
							},
							'computation': {
								'sequence': [],
								'flow_info': CompFlowInfo(
									reg_flows=[],
									block_id=block_id
								)
							},
							'output': {'type': 'register'},
							'loop_context': {
								'level': None,
								'is_reduction': False,
								'is_loop_carried': False
							}
						}

						# ループレベルの設定
						for level, info in self.analyzer.loop_levels.items():
							if block_id in info.nodes:
								current_path['loop_context']['level'] = level
								current_path['computation']['flow_info'].loop_level = int(level)
								break

					# 計算ノードを追加
					self._add_computation_node(current_path, node_idx, node_info, array_access_info)

					# 入力レジスタに関連するロード情報を追加
					for input_reg in node_info[3:]:
						if input_reg.startswith('%') and input_reg in load_info:
							load_data = load_info[input_reg]
							if not any(ld.get('reg') == load_data['reg'] for ld in current_path['inputs']['loads']):
								current_path['inputs']['loads'].append(load_data)

					# パスのタイプを更新
					current_path['type'] = self._update_path_type(current_path['type'], operation)

					# 出力情報を更新
					output_reg = node_info[2]
					current_path['output']['value_reg'] = output_reg

					# 出力レジスタが後でストアされる場合、出力タイプをメモリに変更
					if output_reg in store_info:
						store_data = store_info[output_reg]
						current_path['output'] = {
							'type': 'memory',
							'target_reg': store_data['target_reg'],
							'value_reg': output_reg,
							'array': store_data['array']
						}

					# 出力レジスタが配列アクセスに使用される場合、その情報を追加
					if output_reg in register_to_array:
						current_path['output']['array'] = register_to_array[output_reg]

				# 他の命令の場合は現在のパスを無視
				elif operation not in {'load', 'store', 'getelementptr'} and current_path is not None:
					# 計算パスが途切れたとみなし、パスを追加して新しい計算パスを開始
					paths.append(current_path)
					current_path = None

			# 最後のパスも追加
			if current_path is not None:
				paths.append(current_path)

			# ステップ3: 計算パス間の関連付け
			# 例: 乗算結果が加算の入力として使用される場合
			reg_to_path = {}  # レジスタ → パスのマッピング

			# まず各パスの出力レジスタを記録
			for i, path in enumerate(paths):
				if 'value_reg' in path['output']:
					reg = path['output']['value_reg']
					if reg:
						reg_to_path[reg] = i

			# 次に各パスの入力が他のパスの出力から来ている場合、その情報を追加
			for path in paths:
				for load in path['inputs'].get('loads', []):
					reg = load.get('reg')
					if reg in reg_to_path:
						src_path_idx = reg_to_path[reg]
						src_path = paths[src_path_idx]
						if 'dependencies' not in path:
							path['dependencies'] = []
						path['dependencies'].append({
							'source_path': src_path['path_id'],
							'type': 'register',
							'through': reg
						})

			return paths

		except Exception as e:
			print(f"Error in _identify_block_compute_paths for block {block_id}: {e}")
			import traceback
			traceback.print_exc()
			return paths

	def _get_array_from_reg(self, reg: str) -> Optional[str]:
		"""
		レジスタから配列名を取得
		Args:
			reg: レジスタ文字列（例：'getelementptr_@a_29'や'getelementptr_%25_32'）
		Returns:
			Optional[str]: 配列名('a', 'b', 'c')またはNone
		"""
		try:
			#"print(f"DEBUG: Analyzing reg: {reg}")

			# パターン1: 直接的な配列アクセス (getelementptr_@a_29)
			direct_match = re.search(r'getelementptr_@([abc])_', reg)
			if direct_match:
				array_name = direct_match.group(1)
				#print(f"DEBUG: Found direct array access: {array_name}")
				return array_name

			# パターン2: 中間的なアクセスの場合 (getelementptr_%25_32)
			if reg.startswith('getelementptr_%'):
				#print(f"DEBUG: Checking intermediate access")
				# レジスタ番号を抽出 (例: 25_32から25を取得)
				reg_num = reg.split('_')[1]

				# このレジスタ番号を使用している直接配列アクセスを探す
				for block_id, block_nodes in self.all_nodes.items():
					for node in block_nodes:
						node_info = node[0].split()
						# このレジスタ番号を使用するgetelementptr_@で始まるノードを探す
						if 'getelementptr_@' in str(node_info) and reg_num in str(node_info):
							array_match = re.search(r'@([abc])', str(node_info))
							if array_match:
								array_name = array_match.group(1)
								#print(f"DEBUG: Found array {array_name} for reg {reg_num} in block {block_id}")
								return array_name

			return None

		except Exception as e:
			print(f"Error in _get_array_from_reg: {e}")
			return None

	def _is_valid_operand(self, operand: str) -> bool:
		"""
		オペランドが有効か（レジスタか即値か）を判定
		Args:
			operand: チェックするオペランド
		Returns:
			bool: 有効な場合True
		"""
		try:
			# レジスタ
			if operand.startswith('%'):
				return True

			# 即値（整数）
			if operand.isdigit() or (operand.startswith('-') and operand[1:].isdigit()):
				return True

			# 16進数
			if operand.startswith('0x') or operand.startswith('-0x'):
				try:
					int(operand, 16)
					return True
				except ValueError:
					return False

			return False

		except Exception as e:
			print(f"Error checking operand validity: {e}")
			return False

	def _read_node_list(self, block_id: str) -> List[List[str]]:
		"""
		基本ブロックのノードリストを取得
		すでにAnalyzerクラスに同名メソッドがあるため、それを利用

		Args:
			block_id: 基本ブロックID
		Returns:
			List[List[str]]: ノード情報のリスト
		"""
		try:
			# Analyzerのメソッドを使用
			return self.analyzer._read_node_list(block_id)

		except Exception as e:
			print(f"Error reading node list for block {block_id}: {e}")
			return []

	def _init_compute_path(self, block_id: str, node_id: str, node_info: List[str]) -> Dict:
		try:
			path = {
				'path_id': f"path_{block_id}_{node_id}",
				'type': 'unknown',
				'inputs': {
					'loads': [],
					'leafs': []
				},
				'computation': {
					'sequence': [],
					'flow_info': CompFlowInfo(  # CompFlowInfoを追加
						reg_flows=[],
						block_id=block_id
					)
				},
				'output': {'type': 'register', 'target_reg': node_info[2] if len(node_info) > 2 else None},
				'loop_context': {
					'level': None,
					'is_reduction': False,
					'is_loop_carried': False
				}
			}

			# 既存のループレベル処理
			for level, info in self.analyzer.loop_levels.items():
				if block_id in info.nodes:
					path['loop_context']['level'] = level
					path['computation']['flow_info'].loop_level = int(level)
					break

			operation = node_info[1].split('_')[0]

			# load操作の処理
			if operation == 'load':
				array_name = self.analyzer._get_array_from_reg(node_info[3])
				if array_name:
					load_info = {
						'array': array_name,
						'index_regs': self.analyzer._get_index_regs(node_info[3]),
						'node_id': node_id
					}
					path['inputs']['loads'].append(load_info)
					path['type'] = 'load'

			return path

		except Exception as e:
			print(f"Error initializing compute path: {e}")
			return None

	def _add_computation_node(self, path: Dict, node_id: str, node_info: List[str], array_access_info: Dict) -> None:
		try:
			computation = {
				'node_id': node_id,
				'opcode': node_info[1],
				'input_regs': [],
				'output_reg': node_info[2],
				'array_access': {}
			}

			for operand in node_info[3:]:
				if operand.startswith('%'):
					computation['input_regs'].append(operand)
					if operand in array_access_info:
						computation['array_access'][operand] = array_access_info[operand]

			# RegisterFlowをself経由で使用
			reg_flow = self.RegisterFlow(
				reg=node_info[2],
				opcode=node_info[1].split('_')[0],
				operands=[op for op in node_info[3:] if self._is_valid_operand(op)],
				output=node_info[2],
				extra_info={'array_access': computation['array_access']}
			)

			path['computation']['flow_info'].add_flow(reg_flow)

			# シーケンスへの追加
			sequence = path['computation']['sequence']
			if sequence:
				# 既存の処理
				insert_pos = len(sequence)
				for i, existing in enumerate(sequence):
					if computation['output_reg'] in existing['input_regs']:
						insert_pos = min(insert_pos, i)

				sequence.insert(insert_pos, computation)
			else:
				sequence.append(computation)

			# パスタイプの更新
			operation = node_info[1].split('_')[0]
			path['type'] = self._update_path_type(path['type'], operation)

		except Exception as e:
			print(f"Error adding computation node: {e}")

	def _update_path_type(self, current_type: str, opcode: str) -> str:
		"""計算パスのタイプを更新"""
		try:
			if current_type == 'unknown':
				if 'add' in opcode or 'sub' in opcode:
					return 'add_chain'
				elif 'mul' in opcode:
					return 'multiply'
				elif 'and' in opcode or 'or' in opcode or 'xor' in opcode:
					return 'logical'
				else:
					return 'misc'
			elif current_type == 'multiply' and 'add' in opcode:
				return 'multiply_add'
			# 他の組み合わせの場合は現在のタイプを維持
			return current_type

		except Exception as e:
			print(f"Error updating path type: {e}")
			return 'unknown'

	def _analyze_path_dependencies(self) -> List[Dict]:
		"""パス間の依存関係を分析"""
		try:
			dependencies = []

			# パスのペアごとに依存関係を確認
			for i, path1 in enumerate(self.compute_paths):
				for path2 in self.compute_paths[i+1:]:
					# レジスタ依存の確認
					reg_deps = self._find_register_dependency(path1, path2)
					dependencies.extend(reg_deps)

					# メモリ依存の確認
					mem_deps = self._find_memory_dependency(path1, path2)
					dependencies.extend(mem_deps)

					# ループ伝搬依存の確認
					loop_deps = self._find_loop_carried_dependency(path1, path2)
					dependencies.extend(loop_deps)

			return dependencies

		except Exception as e:
			print(f"Error analyzing path dependencies: {e}")
			return []

	def _find_register_dependency(self, path1: Dict, path2: Dict) -> List[Dict]:
		"""レジスタ依存の検出"""
		try:
			deps = []

			# path1の出力レジスタ
			output_regs = set()
			for comp in path1.get('computation', {}).get('sequence', []):
				output_regs.add(comp['output_reg'])

			# path2の入力レジスタ
			input_regs = set()
			for comp in path2.get('computation', {}).get('sequence', []):
				input_regs.update(comp['input_regs'])

			# 依存関係の検出
			for reg in output_regs & input_regs:  # 共通するレジスタを検出
				deps.append({
					'source_path': path1['path_id'],
					'target_path': path2['path_id'],
					'type': 'register',
					'through': reg
				})

			return deps

		except Exception as e:
			print(f"Error finding register dependency: {e}")
			return []

	def _find_memory_dependency(self, path1: Dict, path2: Dict) -> List[Dict]:
		"""メモリ依存の検出"""
		try:
			deps = []

			# path1の出力配列アクセス
			outputs1 = []
			if path1.get('output', {}).get('type') == 'memory':
				outputs1.append(path1['output'])

			# path2の入力配列アクセス
			inputs2 = path2.get('inputs', {}).get('loads', [])

			# 依存関係の検出
			for out in outputs1:
				for inp in inputs2:
					if out['array'] == inp['array']:
						# 同じ配列へのアクセスを検出
						if self._check_array_indices_overlap(
							out.get('index_regs', []),
							inp.get('index_regs', [])
						):
							deps.append({
								'source_path': path1['path_id'],
								'target_path': path2['path_id'],
								'type': 'memory',
								'through': out['array']
							})

			return deps

		except Exception as e:
			print(f"Error finding memory dependency: {e}")
			return []

	def _find_loop_carried_dependency(self, path1: Dict, path2: Dict) -> List[Dict]:
		"""ループ伝搬依存の検出"""
		try:
			deps = []

			# 両パスのループコンテキストを取得
			loop_ctx1 = path1.get('loop_context', {})
			loop_ctx2 = path2.get('loop_context', {})

			# 同じループレベルに属しているか確認
			if (loop_ctx1.get('level') == loop_ctx2.get('level') and
				loop_ctx1.get('is_loop_carried') and
				loop_ctx2.get('is_loop_carried')):

				# レジスタ依存のチェック
				reg_deps = self._find_register_dependency(path1, path2)
				for dep in reg_deps:
					deps.append({
						**dep,
						'type': 'loop_carried',
						'loop_level': loop_ctx1['level']
					})

				# メモリ依存のチェック
				mem_deps = self._find_memory_dependency(path1, path2)
				for dep in mem_deps:
					deps.append({
						**dep,
						'type': 'loop_carried',
						'loop_level': loop_ctx1['level']
					})

			return deps

		except Exception as e:
			print(f"Error finding loop-carried dependency: {e}")
			return []

	def _check_array_indices_overlap(self, indices1: List[str], indices2: List[str]) -> bool:
		"""配列インデックスの重なりをチェック"""
		try:
			# インデックスレジスタが完全に一致する場合
			if set(indices1) == set(indices2):
				return True

			# インデックス計算式の解析が必要な場合
			common_indices = set(indices1) & set(indices2)
			return len(common_indices) > 0

		except Exception as e:
			print(f"Error checking array indices overlap: {e}")
			return False

class Analyzer:
	def __init__(self, r_file_path: str, r_name: str):
		#print("\nDEBUG: Initializing Analyzer")
		self.r_path = r_file_path
		self.r_name = r_name
		self.loops = self._read_loop_structure()
		self.array_patterns = {'array_patterns': {}}
		self.pointer_regs_info = {}
		self.store_load_deps = {}
		self.cfg_connectivity = {}
		self._node_list_cache = {}
		self._am_cache = {}
		self._gep_chain_cache = {}
		self.nodes_cache = {}
		all_nodes, node_to_block = self._collect_all_nodes()
		self.all_nodes = all_nodes
		self.node_to_block = node_to_block
		self.loop_levels = self._analyze_loop_levels()
		self.compute_path_analyzer = ComputeDataPath(self)
		#print("  Setting up array_dims")
		self.array_dims = self._get_array_dimensions()
		#print(f"  array_dims initialized: {self.array_dims}")

	def _read_path_file(self, block_id: str, path_type: str) -> List[str]:
		"""パスファイルの読み込み"""
		try:
			path_file = os.path.join(self.r_path,
				#REMOVE
				#f"{self.r_name}_bblock_{block_id}_bpath_{path_type}.txt")
				f"noundef_bblock_{block_id}_bpath_{path_type}.txt")
			if not os.path.exists(path_file):
				return []
			with open(path_file, 'r') as f:
				return f.readlines()
		except Exception:
			return []

	@functools.lru_cache(maxsize=128)
	def _read_node_list(self, block_id: str) -> List[List[str]]:
		"""
		基本ブロックのノードリストを読み込み
		"""
		if block_id in self.nodes_cache:
			return self.nodes_cache[block_id]

		try:
			file_path = os.path.join(self.r_path, f"noundef_bblock_{block_id}_node_list.txt")
			if not os.path.exists(file_path):
				print(f"Warning: Node file not found: {file_path}")
				return []

			nodes: List[List[str]] = []
			with open(file_path, 'r') as f:
				for line in f:
					node = line.strip().split(',')
					nodes.append(node)

			# デバッグ出力を追加
			#if block_id in ['5', '9', '19']:  # ループヘッダーブロックのみ出力
			#	print(f"\nDEBUG: Contents of block {block_id} (loop header)")
			#	for node in nodes:
			#		node_info = node[0].split()
			#		if len(node_info) > 1:
			#			# 重要な命令（分岐、比較、GEP）のみ出力
			#			if any(op in node_info[1] for op in ['br', 'icmp', 'getelementptr']):
			#				print(f"  {' '.join(node_info)}")

			self.nodes_cache[block_id] = nodes
			return nodes

		except Exception as e:
			print(f"Error reading node list from {file_path}: {e}")
			return []

	def _read_loop_structure(self) -> List[List[str]]:
		"""
		ループ構造情報の読み込みと解析

		Returns:
			List[List[str]]: [
				[node_id, ...],  # 最内ループ
				[node_id, ...],  # 中間ループ
				[node_id, ...],  # 最外ループ
			]
			内側から外側のループの順でノードIDのリストを返す
		"""
		try:
			# 1. ループ構造ファイルの読み込み
			#REMOVE
			#loop_file_path = os.path.join(self.r_path, f"{self.r_name}_cfg_loop.txt")
			loop_file_path = os.path.join(self.r_path, f"noundef_cfg_loop.txt")
			if not os.path.exists(loop_file_path):
				return []

			with open(loop_file_path, 'r') as f:
				content = f.read().strip()
				if not content:
					return []

			# 2. ループ構造の解析
			content = content.strip('[]')
			loop_strs = content.split('],')
			loops = []

			# 3. 各ループの処理
			for loop_str in loop_strs:
				loop_str = loop_str.replace('[', '').replace(']', '')
				nodes = []

				# ループノードの抽出
				for node in loop_str.split(','):
					node = node.strip().strip("'").strip('"')
					if node:  # 空のノードは除外
						nodes.append(node)

				if nodes:  # 空のループは除外
					loops.append(nodes)

			# 4. ループの検証
			validated_loops = []
			for loop in loops:
				# 各ループが最低2つのノードを持つことを確認
				if len(loop) >= 2:
					validated_loops.append(loop)
				else:
					print(f"Warning: Invalid loop structure detected: {loop}")

			return validated_loops

		except Exception as e:
			print(f"Error reading loop structure: {e}")
			return []

	def _path_formatter(self, paths: List[str]) -> List[List[List[str]]]:
		"""
		パス情報のフォーマット処理
		Args:
			paths: パス情報の文字列リスト
				例: ["[1,2,3][4,5,6]", "[7,8,9][10,11,12]"]
		Returns:
			List[List[List[str]]]: [
				[["1","2","3"], ["4","5","6"]],  # 1つ目のパス
				[["7","8","9"], ["10","11","12"]] # 2つ目のパス
			]
		"""
		try:
			formatted_paths = []

			for path in paths:
				if not path.strip():
					continue

				current_path = []
				path_segments = path.split(']')

				for segment in path_segments:
					if not segment.strip():
						continue

					node_ids = segment.replace('[', '').strip()
					if node_ids:
						nodes = [n.strip() for n in node_ids.split(',')]
						nodes = [n for n in nodes if n]
						if nodes:
							current_path.append(nodes)

				if current_path:
					formatted_paths.append(current_path)

			return formatted_paths

		except Exception as e:
			print(f"Error in path formatter: {e}")
			return []

	def _path_formatter2(self, path: str) -> List[List[str]]:
			"""
			パス情報のフォーマット処理 (文字列またはリストに対応)
			"""
			try:
				formatted_path = []
				if not path:
					return formatted_path

				if isinstance(path, str):
					segments = path.split('[')[1:]

					for segment in segments:
						node_ids = segment.replace(']', '').replace(' ', '')
						if node_ids:
							nodes = node_ids.split(',')
							nodes = [n for n in nodes if n]
							if nodes:
								formatted_path.append(nodes)
				elif isinstance(path, list):
					formatted_path.append([str(n) for n in path])
				else:
					print(f"Warning: Invalid path type: {type(path)}")
					return []

				return formatted_path

			except Exception as e:
				print(f"Error in path formatter2: {e}")
				return []

	def _format_path(path_str: Optional[str]) -> List[List[str]]:
		"""
		Formats a path string into a list of node ID lists.

		Handles both single and multiple path strings.

		Args:
			path_str: The path string to format (e.g., "[1,2,3][4,5,6]" or "[[1,2,3][4,5,6]][[7,8][9]]").
				Can be None or empty.

		Returns:
			A list of lists of node IDs, or an empty list if the input is invalid.
			For example:
				_format_path("[1,2,3][4,5,6]") == [["1", "2", "3"], ["4", "5", "6"]]
				_format_path("[[1,2][3]][[4,5]]") == [["1","2"], ["3"]], [["4","5"]]
				_format_path("") == []
				_format_path(None) == []
		"""
		if not path_str or not path_str.strip():
			return []

		paths = path_str.split('][')
		formatted_paths: List[List[List[str]]] = []
		for path in paths:
			path = path.strip('[]')

			formatted_path: List[List[str]] = []
			segments = re.findall(r"\[([^\]]+)\]", path)
			for segment in segments:
				nodes = [node.strip() for node in segment.split(',') if node.strip()]
				if nodes:
					formatted_path.append(nodes)
			if formatted_path:
				formatted_paths.append(formatted_path)

		if len(formatted_paths) == 1:
			return formatted_paths[0]
		return formatted_paths

	def _determine_index_dimension(self, node: List[str], index_position: int) -> int:
		"""
		getelementptrのインデックスが何次元目のアクセスに使用されているか判断
		Args:
			node: getelementptrノードの情報
			index_position: インデックスの位置
		Returns:
			次元番号（0から開始）または-1（エラー時）
		"""
		try:
			# getelementptrの型情報からインデックスの次元を判断
			dimension_count = 0
			for i, token in enumerate(node):
				if token.startswith('%') and token != '%0':
					if i == index_position:
						return dimension_count
					dimension_count += 1
			return -1

		except Exception as e:
			print(f"Error determining index dimension: {e}")
			return -1

	def _detect_array_access(self, block_id, pointer_regs):
		"""
		ブロック内の配列アクセスを検出

		Args:
			block_id: ブロックID
			pointer_regs: ポインタレジスタ情報

		Returns:
			{
				'mem_ops': {
					'loads': [load_info, ...],
					'stores': [store_info, ...]
				},
				'store_deps': {
					store_line_num: {
						'loads': [load_info, ...]
					}
				}
			} or None
		"""
		try:
			nodes = self._read_node_list(block_id)
			if not nodes:
				return None

			mem_ops = {'loads': [], 'stores': []}
			store_deps = {}

			for line_num, node in enumerate(nodes):
				node = node[0].split()
				if len(node) > 1:
					if 'load' in node[1]:
						for pointer_reg in pointer_regs:
							if pointer_reg["block_id"] == block_id:
								for index_reg in pointer_reg["index_regs"]["regs"]:
									if index_reg in str(node):
										mem_ops['loads'].append({'line_num': line_num})
										break
					elif 'store' in node[1]:
						for pointer_reg in pointer_regs:
							if pointer_reg["block_id"] == block_id:
								for index_reg in pointer_reg["index_regs"]["regs"]:
									if index_reg in str(node):
										mem_ops['stores'].append({'line_num': line_num, 'value': node[2]})
										break

			if mem_ops['loads'] or mem_ops['stores']:
				return {'mem_ops': mem_ops, 'store_deps': store_deps}
			else:
				return None

		except Exception as e:
			print(f"Error detecting array access in block {block_id}: {e}")
			return None

	def _paths_equal(self, path1, path2):
		"""パスが等しいかどうかを判定"""
		if len(path1['blocks']) != len(path2['blocks']):
			return False
		return path1['blocks'] == path2['blocks']

	def _find_begin_geps(self, block_id: str) -> List[Dict[str, Any]]:
		"""
		始端getelementptrノードの特定
		Args:
			block_id: 基本ブロックID
		Returns:
			[
				{
					'array_name': str,     # 配列名
					'path_no': int,        # パス番号
					'gep_node_id': str,    # GEPノードID
					'begin_node_id': str   # 始端ノードID
				}
			]
		デフォルト値：[{"array_name": "", "path_no": 0, "gep_node_id": "0", "begin_node_id": "0"}]
		"""
		begin_geps = []
		default_gep = {
			"array_name": "",
			"path_no": 0,
			"gep_node_id": "0",
			"begin_node_id": "0"
		}

		try:
			# 1. ノード情報の取得
			nodes = self._read_node_list(block_id)
			if not nodes:
				return [default_gep]

			# 2. ld-to-ldパスファイルの読み込みと解析
			ld_ld_path = os.path.join(self.r_path,
				#REMOVE
				#f"{self.r_name}_bblock_{block_id}_bpath_ld_ld.txt")
				f"noundef_bblock_{block_id}_bpath_ld_ld.txt")

			if not os.path.exists(ld_ld_path):
				return [default_gep]

			with open(ld_ld_path, 'r') as f:
				ld_ld_paths = f.readlines()

			if not ld_ld_paths:
				return [default_gep]

			# 3. パス情報の解析
			paths = self._path_formatter(ld_ld_paths)[0]
			if not paths or not paths[0]:
				return [default_gep]

			# 4. 各パスの解析
			for list_no, path in enumerate(paths):
				if not path:
					continue

				begin_node_id = path[-1].strip()
				if not begin_node_id:
					continue

				# 5. パス内のGEPノード探索
				BREAK = False
				for node_id in reversed(path):
					node_id = node_id.strip()
					if not node_id or not node_id.isdigit():
						continue

					node_idx = int(node_id)
					if node_idx >= len(nodes):
						continue

					# 6. getelementptrノードの判定
					node = nodes[node_idx][0]
					if len(node) < 1:
						continue

					opcode = node.split()[1]
					if 'getelementptr' in opcode:
						array_name = None

						# 形式1, 2に対応 (@<array_name>)
						match = re.search(r"@(\w+)", node)
						if match:
							array_name = match.group(1).split('_')[0]

						# 形式3に対応 (getelementptr_<array_name>)
						if array_name is None:  # @形式で見つからなかった場合
							match = re.search(r"getelementptr_([a-zA-Z0-9_]+)", node)
							if match:
								array_name = match.group(1).split('_')[0]

						# 配列名が見つからなくても情報を保持
						begin_geps.append({
							"array_name": array_name,	# None の場合もあり
							"path_no": list_no,
							"gep_node_id": node_id,
							"begin_node_id": begin_node_id
						})
						break

			return begin_geps if begin_geps else [default_gep]

		except Exception as e:
			print(f"Error in _find_begin_geps for block {block_id}: {e}")
			return [default_gep]

	def _find_terminal_geps(self,
			block_id: str,
			gep_nodes: List[Tuple[int, List[str]]]) -> List[Tuple[int, List[str]]]:
		"""
		終端getelementptrノードの特定
		Args:
			block_id: 基本ブロックID
			gep_nodes: [(行番号, ノード情報), ...] 形式のgetelementptrノードリスト
		Returns:
			[(行番号, ノード情報), ...] 形式の終端GEPノードのリスト
		"""
		terminal_geps = []

		try:
			# 1. ノードリストの読み込み
			nodes = self._read_node_list(block_id)
			if not nodes:
				return terminal_geps

			# 2. AMファイルの読み込みと処理
			#REMOVE
			#am_file = f"{self.r_name}_bblock_{block_id}"
			am_file = f"noundef_bblock_{block_id}"
			am_size, am = AMUtils.Preprocess(self.r_path, am_file)

			# 3. 各getelementptrノードの解析
			for gep in gep_nodes:
				gep_line = gep[0]
				# 接続先を探索
				for dst_idx in range(am_size):
					if am[gep_line][dst_idx]:
						# 接続先のノードを確認
						if dst_idx >= len(nodes):
							continue

						dst_node = nodes[dst_idx][0].split()
						if len(dst_node) > 1:
							# load/store命令に接続している場合は終端GEPと判定
							if 'load' in dst_node[1] or 'store' in dst_node[1]:
								terminal_geps.append(gep)
								break

							# 別のGEPに接続している場合はチェーンの一部
							if 'getelementptr' in dst_node[1]:
								continue

			return terminal_geps

		except Exception as e:
			print(f"Error in _find_terminal_geps for block {block_id}: {e}")
			print(f"Context - GEP nodes: {len(gep_nodes)}")
			return terminal_geps

	def _find_forward_loads(self,
			target_block: str,
			store_reg: str,
			store_line: int,
			source_block: str,
			loop_nodes: List[str]) -> Dict:
		"""
		CFG順方向でのload命令検索
		Args:
			target_block: 検索対象ブロック
			store_reg: 検索対象レジスタ
			store_line: store命令の行番号
			source_block: ソースブロック
			loop_nodes: ループノード群
		Returns:
			{
				'loads': [
					{
						'line_num': int,
						'is_loop_edge': bool,
						'edge_type': str
					}
				]
			}
		"""
		result = {'loads': []}

		try:
			# 1. ターゲットブロックのノード情報取得
			node_info = self._read_node_list(target_block)
			if not node_info:
				return result

			# 2. ループエッジの判定
			is_loop_edge = False
			edge_type = 'normal'
			if source_block == loop_nodes[0] and target_block == loop_nodes[-1]:
				is_loop_edge = True
				edge_type = 'loop_forward'
			elif source_block == loop_nodes[-1] and target_block == loop_nodes[0]:
				is_loop_edge = True
				edge_type = 'loop_back'

			# 3. load命令の検索
			for line_num, node in enumerate(node_info):
				node = node[0].split()
				if len(node) < 2:
					continue
				if 'load' in node[1] and store_reg in node[3]:
					load_info = {
						'line_num': line_num,
						'is_loop_edge': is_loop_edge,
						'edge_type': edge_type
					}
					result['loads'].append(load_info)

			return result

		except Exception as e:
			print(f"Error finding forward loads in block {target_block}: {e}")
			return result

	def _is_loop_carried_register(self, reg: str, block_id: str, loop_info: Dict) -> bool:
		"""
		レジスタがループ伝搬依存を持つか判定
		Args:
			reg: 検査対象レジスタ
			block_id: 基本ブロックID
			loop_info: ループ構造情報
		Returns:
			bool: ループ伝搬依存の有無
		"""
		try:
			if not loop_info['edges']['loop_carried']:
				return False

			nodes = self._read_node_list(block_id)
			if not nodes:
				return False

			# レジスタの定義位置を確認
			for line_num, node in enumerate(nodes):
				node - node[0].split()
				if len(node) > 2 and reg in node[1]:
					# レジスタがループヘッダで定義され、ループ内で使用される場合
					if (loop_info['loop_info']['current']['is_header'] and
						any(self._is_reg_used_in_block(reg, loop_node)
							for loop_node in loop_info['loop_info']['nodes'])):
						return True

			return False

		except Exception as e:
			print(f"Error checking loop carried register {reg}: {e}")
			return False

	def _is_reg_used_in_block(self, reg: str, block_id: str) -> bool:
		"""
		指定したブロック内でレジスタが使用されているか確認
		"""
		try:
			nodes = self._read_node_list(block_id)
			return any(reg in node[0].split() for node in nodes if len(node) > 1)
		except Exception:
			return False

	def _get_index_regs(self, reg: str) -> List[str]:
		"""
		レジスタから配列インデックスに使用されるレジスタ情報を取得
		Args:
			reg: レジスタ名 (getelementptr形式または@シンボル形式)
		Returns:
			List[str]: インデックスレジスタのリスト
		"""
		try:
			index_regs = []

			# 全基本ブロックのノードを探索
			for block_id, nodes in self.all_nodes.items():
				for node in nodes:
					node_info = node[0].split()
					if 'getelementptr' in str(node_info[1]):
						# このregを使用するgetelementptrを探す
						if reg in node_info[2:]:
							# レジスタ形式のオペランドを収集
							for operand in node_info[2:]:
								if operand.startswith('%'):
									if operand not in index_regs:
										index_regs.append(operand)

			return index_regs

		except Exception as e:
			print(f"Error getting index registers from {reg}: {e}")
			return []

	def _get_dependency_type(self, opcode: str) -> str:
		"""オペコードから依存関係の種類を判定"""
		if 'load' in opcode:
			return 'load'
		elif 'phi' in opcode:
			return 'phi'
		return 'calc'

	def _get_all_block_ids(self):
		"""全ての基本ブロックIDを取得する"""
		try:
			#REMOVE
			#cfg_loop_file = os.path.join(self.r_path, f"{self.r_name}_cfg_loop.txt")
			cfg_loop_file = os.path.join(self.r_path, f"noundef_cfg_loop.txt")
			if not os.path.exists(cfg_loop_file):
				print(f"Warning: CFG loop file not found: {cfg_loop_file}")
				return []

			block_ids = []
			with open(cfg_loop_file, 'r') as f:
				import ast
				loops = ast.literal_eval(f.read())
				block_ids = list(set([block_id for loop in loops for block_id in loop]))
			return block_ids

		except FileNotFoundError:
			print(f"Error: CFG loop file not found: {cfg_loop_file}")
			return []
		except (SyntaxError, ValueError) as e:
			print(f"Error parsing CFG loop file: {e}")
			return []
		except Exception as e:
			print(f"An unexpected error occurred: {e}")
			return []

	def _build_forward_path(self, block_id, array_access, store_load_deps, cfg_connectivity, current_path, access_paths):
		"""
		配列アクセスの前方パスを構築

		Args:
			block_id: 現在のブロックID
			array_access: 配列アクセス情報
			store_load_deps: Store-Load依存関係
			cfg_connectivity: CFG接続情報
			current_path: 現在のパス情報
			access_paths: すべてのアクセスパス
		"""
		try:
			# 現在のブロックをパスに追加
			current_path['blocks'].append(block_id)

			# メモリオペレーションの追加
			if 'mem_ops' not in current_path:
				current_path['mem_ops'] = {}
			if block_id not in current_path['mem_ops']:
				current_path['mem_ops'][block_id] = {}
			current_path['mem_ops'][block_id].update(array_access['mem_ops'])

			# Store依存の追加
			if 'store_deps' not in current_path:
				current_path['store_deps'] = {}
			if block_id not in current_path['store_deps']:
				current_path['store_deps'][block_id] = {}
			current_path['store_deps'][block_id].update(array_access.get('store_deps', {}))

			# 次のブロックの探索
			next_blocks = cfg_connectivity['forward_edges'].get(block_id, [])

			if not next_blocks:  # 末端ブロックの場合
				# パスの重複チェック
				if not any(self._paths_equal(current_path, existing_path) for existing_path in access_paths):
					access_paths.append(current_path.copy())  # パスを追加
				return

			for next_block_info in next_blocks:
				next_block = next_block_info['target']
				is_loop_edge = next_block_info['is_loop_edge']

				# ループバックエッジの場合、パスを打ち切る
				if is_loop_edge and next_block in current_path['blocks']:
					if not any(self._paths_equal(current_path, existing_path) for existing_path in access_paths):
						current_path["is_loop_path"] = True #Set loop path
						access_paths.append(current_path.copy())
					continue

				# 次のブロックの情報を取得
				next_block_info = next((info for info in store_load_deps.get('block_info', []) if info['block_id'] == next_block), None)
				if not next_block_info:
					continue

				next_array_access = self._detect_array_access(next_block, next_block_info)
				if not next_array_access:
					continue

				#IndexExpressionの引き継ぎ
				next_path = current_path.copy()
				next_path["index_expressions"] = current_path["index_expressions"].copy()

				# 再帰的にパスを構築
				self._build_forward_path(
					next_block,
					next_array_access,
					store_load_deps,
					cfg_connectivity,
					next_path,
					access_paths
				)

		except Exception as e:
			print(f"Error building forward path: {e}")
			print(f"Context - Block: {block_id}")

	def _trace_index_variable(self, index_reg, start_block, paths, cfg_connectivity, loop_blocks):
		"""インデックス変数の追跡を改善"""
		#print(f"\nDEBUG: Tracing index register {index_reg} starting from block {start_block}")
		#print(f"  Loop blocks: {loop_blocks}")

		# ループインデックス変数を特定
		loop_indices = {}
		for level, loop_info in self.loop_levels.items():
			loop_index = self._identify_loop_index_var(loop_info.header)
			if loop_index:
				loop_indices[level] = loop_index
				print(f"  Loop level {level} index: {loop_index}")

		trace_info = {
			'path': [],
			'target_reg': index_reg,
			'leaf_node': None,
			'derived_from_loop': None  # 追加: どのループインデックスから派生したか
		}

		# 追跡管理
		tracked_nodes = set()
		tracked_regs = set([index_reg])
		#print(f"  Initial tracked registers: {tracked_regs}")

		# すべてのループブロックを探索
		for block_id in loop_blocks:
			print(f"  Searching in block {block_id}")
			nodes = self._read_node_list(block_id)
			related_nodes = []

			# このブロックでレジスタに関連するノードを探す
			for node_idx, node in enumerate(nodes):
				node_info = node[0].split()
				if (block_id, node_idx) in tracked_nodes:
					continue

				# レジスタが使用または定義されているか確認
				reg_involved = False
				reg_position = -1
				for i, part in enumerate(node_info):
					if part in tracked_regs:
						reg_involved = True
						reg_position = i
						break

				if reg_involved:
					tracked_nodes.add((block_id, node_idx))

					opcode = node_info[1].split('_')[0] if len(node_info) > 1 else 'unknown'
					if 'LEAF' in node_info:
						opcode = 'LEAF'

					# レジスタフロー情報を作成
					flow = RegisterFlow(
						reg=index_reg,
						opcode=opcode,
						operands=[op for op in node_info[3:] if op.startswith('%')],
						output=node_info[2] if len(node_info) > 2 and node_info[2].startswith('%') else '',
						extra_info={
							'block_id': block_id,
							'node_id': node_idx,
							'instruction': ' '.join(node_info)
						}
					)

					related_nodes.append(flow)

					# ループインデックスとの関連を確認
					for i, part in enumerate(node_info):
						if i >= 3:  # オペランド部分をチェック
							# ループインデックス変数との直接の関連
							for level, loop_idx in loop_indices.items():
								if part == loop_idx:
									#print(f"  Register {index_reg} directly related to loop index {loop_idx} (level {level})")
									if not trace_info['derived_from_loop']:
										trace_info['derived_from_loop'] = {
											'level': level,
											'index_reg': loop_idx,
											'relation': 'direct',
											'node_info': ' '.join(node_info)
										}

					# 新しいレジスタを追跡対象に追加
					if len(node_info) > 2 and node_info[2].startswith('%'):
						tracked_regs.add(node_info[2])

					# LEAFノードを特定
					if opcode == 'LEAF':
						trace_info['leaf_node'] = flow

			if related_nodes:
				#print(f"  Found {len(related_nodes)} related nodes in block {block_id}")
				new_regs = set()
				for node in related_nodes:
					if hasattr(node, 'output') and node.output.startswith('%'):
						new_regs.add(node.output)
				if new_regs:
					#print(f"  New registers found: {new_regs}")
					tracked_regs.update(new_regs)
				trace_info['path'].extend(related_nodes)

		# LEAF ノードが見つからなかった場合は最後のノードを使用
		if not trace_info['leaf_node'] and trace_info['path']:
			trace_info['leaf_node'] = trace_info['path'][-1]

		# ループインデックスとの関連性を確認 (間接的な関係を探る)
		if not trace_info['derived_from_loop'] and trace_info['path']:
			# パス内の全ノードで使用されているオペランドをチェック
			all_operands = set()
			for node in trace_info['path']:
				if hasattr(node, 'operands'):
					all_operands.update(node.operands)

			# ループインデックスとの間接的な関連
			for operand in all_operands:
				for level, loop_idx in loop_indices.items():
					# 簡易的なチェック: オペランドからループインデックスまでの派生関係
					# 本来はより詳細な解析が必要
					if self._is_reg_derived_from(operand, loop_idx, start_block):
						#print(f"  Register {index_reg} indirectly related to loop index {loop_idx} (level {level}) via {operand}")
						if not trace_info['derived_from_loop']:
							trace_info['derived_from_loop'] = {
								'level': level,
								'index_reg': loop_idx,
								'relation': 'indirect',
								'via_reg': operand
							}

		# 結果の詳細を出力
		#if trace_info['derived_from_loop']:
		#	print(f"  Trace result: Index register {index_reg} derived from loop level {trace_info['derived_from_loop']['level']}")
		#else:
		#	print(f"  Trace result: No loop index relation found for {index_reg}")

		#print(f"  Total nodes in path: {len(trace_info['path'])}")
		return trace_info if trace_info['path'] else None

	def _collect_all_nodes(self):
			"""全ての基本ブロックのノード情報を収集し、node_to_blockも作成"""
			all_nodes = {}
			node_to_block = {}
			try:
				block_ids = self._get_all_block_ids()

				for block_id in block_ids:
					nodes = self._read_node_list(block_id)
					if nodes is None:
						continue
					all_nodes[block_id] = nodes
					for node in nodes:
						node = node[0].split()
						if node and node[0].isdigit():
							node_to_block[node[0]] = block_id

				return all_nodes, node_to_block
			except Exception as e:
				print(f"Error collecting all nodes: {e}")
				return {}

	def _collect_index_registers(self,
			block_id: str,
			term_gep,
			begin_geps) -> Dict[str, Union[List[str], int]]:
		"""インデックスレジスタの収集"""
		reg_info = {
			'regs': [],
			'array_dim': 0,
			'dependencies': {}
		}
		array_name = 'None'
		try:
			# 1. パスファイルの読み込み
			paths = {
				'ld_leaf': self._read_path_file(block_id, 'ld_leaf'),
				'ld_ld': self._read_path_file(block_id, 'ld_ld')
			}

			if not any(paths.values()) or not paths['ld_ld']:
				return array_name, reg_info

			# 2. ノード情報の取得
			nodes = self._read_node_list(block_id)
			if not nodes:
				return array_name, reg_info

			# 3. AMの読み込み
			#REMOVE
			#am_file = f"{self.r_name}_bblock_{block_id}"
			am_file = f"noundef_bblock_{block_id}"
			if am_file not in self._am_cache:
				am_size, am = AMUtils.Preprocess(self.r_path, am_file)
				self._am_cache[am_file] = (am_size, am)
			else:
				am_size, am = self._am_cache[am_file]

			# 4. 終端ノードからのレジスタ収集
			formatted_paths = self._path_formatter(paths['ld_ld'])
			if not formatted_paths:
				return array_name, reg_info

			array_name, term_registers, ld_node_ids = self._collect_from_terminal(
				term_gep[0],
				formatted_paths[0],
				nodes
			)
			if term_registers:
				for reg in term_registers:
					if reg not in reg_info['regs']:
						reg_info['regs'].append(reg)

			# 5. 始端ノードからのレジスタ収集
			begin_registers = []
			regs = self._collect_from_begin(ld_node_ids, paths['ld_leaf'], nodes)
			begin_registers.extend(regs)
			if begin_registers:
				for reg in begin_registers:
					if reg not in reg_info['regs']:
						reg_info['regs'].append(reg)

			# 6. 依存関係の解析
			deps = {}
			for reg in reg_info['regs']:
				dep_info = self._analyze_register_dependency(
					reg, nodes, am, am_size)
				if dep_info:
					deps[reg] = dep_info
			reg_info['dependencies'] = deps

			# 7. 次元数の設定
			reg_info['array_dim'] = len(reg_info['regs'])

			return array_name, reg_info

		except Exception as e:
			print(f"Error collecting index registers: {e}")
			return array_name, {'regs': [], 'array_dim': 0, 'dependencies': {}}

	def _collect_from_begin(self, ld_node_ids, leaf_paths, nodes):
		"""始端ノードからのレジスタ収集"""
		registers = []
		try:
			for ld_node_id in ld_node_ids:
				for leaf_path_str in leaf_paths:
					leaf_paths = self._path_formatter2(leaf_path_str)
					for leaf_path in leaf_paths:
						if str(ld_node_id) in str(leaf_path[0]):
							leaf_node = nodes[int(leaf_path[-1])][0].split()
							if leaf_node[-1] == 'LEAF' and leaf_node[1].startswith('%'):
								reg = leaf_node[1]
								if reg not in registers:
									registers.append(reg)
			return registers

		except Exception:
			return registers

	def _collect_from_terminal(self, term_node_id: str, ld_paths: List[str], nodes: List[List[str]]) -> List[str]:
			"""終端ノードからのレジスタ収集"""
			registers = []
			start_node_ids = []
			array_name = ''
			path_no = []
			try:
				for no, ld_path in enumerate(ld_paths):
					start_node_id = ld_path[-1]

					if str(term_node_id) not in ld_path:
						continue

					start_node_ids.append(start_node_id)
					path_no.append(no)
					for node in nodes:
						node = node[0].split()

						if len(node) > 2 and str(term_node_id) in node[0] and 'load' in node[1]:
							reg = node[-1]
							if reg.startswith('%') and reg not in registers:
								registers.append(reg)
				for no in path_no:
					ld_path = ld_paths[no]
					for id in ld_path:
						node = nodes[int(id)]
						node = node[0].split()
						if '@' in node[1]:
							array_name = node[1].split('_')[1][1:]

				return array_name, registers, start_node_ids

			except Exception as e:
				print(f"Error in _collect_from_terminal: {e}")
			return array_name, registers, start_node_ids

	def _analyze_array_paths(self, pointer_regs_info, store_load_deps, cfg_connectivity):
		result = {}

		try:
			block_order = store_load_deps.get('block_order', {})
			if not block_order:
				return result

			for loop_level, blocks in block_order.items():
				loop_blocks = blocks.get('sequence', [])
				start_node = blocks.get('start', '')
				end_node = blocks.get('end', '')

				if not loop_blocks or not start_node or not end_node:
					continue

				for block_id in loop_blocks:
					for pointer_regs in pointer_regs_info:
						if pointer_regs.get('block_id') != block_id:
							continue

						array_name = pointer_regs.get('array_name')
						if not array_name:
							continue

						if array_name not in result:
							result[array_name] = {'access_paths': []}

						array_access = self._detect_array_access(block_id, pointer_regs)
						if not array_access:
							continue

						index_regs = pointer_regs['index_regs']['regs']
						index_expressions = []
						for index_reg in index_regs:
							path = self._trace_index_variable(index_reg, block_id, store_load_deps.get("paths"), self.nodes)
							if path:
								index_expression = IndexExpression(
									base=array_name,
									path=path,
									source_variables=[index_reg],
									loop_level=int(loop_level),
									is_loop_carried=self._is_loop_carried_register(index_reg, block_id, self._analyze_loop_structure(block_id))
								)
								index_expressions.append(index_expression)

						current_path = {
							'blocks': [],
							'index_expressions': index_expressions,
							'mem_ops': {},
							'store_deps': {},
							'is_loop_path': False,
							'loop_level': int(loop_level),
							'loop_position': {
								'start': start_node,
								'end': end_node
							}
						}

						self._build_forward_path(
							block_id,
							array_access,
							store_load_deps,
							cfg_connectivity,
							current_path,
							result[array_name]['access_paths']
						)

			return result

		except Exception as e:
			print(f"Error analyzing array paths: {e}")
			return result

	# ループヘッダーブロックからインデックス変数を特定
	def _identify_loop_index_var(self, header_block: str) -> str:
		"""ループヘッダーブロックからインデックス変数を特定"""
		nodes = self._read_node_list(header_block)
		for node in nodes:
			node_info = node[0].split()
			if 'icmp' in node_info[1]:  # 比較命令を探す
				# 比較の左オペランドがループインデックス
				if len(node_info) > 4 and node_info[3].startswith('%'):
					return node_info[3]
		return None

	# GEP命令で使用されるインデックスがどのループレベル由来かを特定
	def _trace_index_to_loop_level(self, index_reg: str) -> Dict[str, str]:
		"""
		インデックスレジスタがどのループレベルのインデックスから派生したかを追跡
		"""
		#print(f"\nDEBUG: Tracing index register {index_reg} to loop level")

		result = {}

		# 1. 各ループレベルのソースインデックス変数を特定
		loop_sources = {}  # {level: (memory_var, loop_counter)}
		for level, loop_info in self.loop_levels.items():
			header_block = loop_info.header
			nodes = self._read_node_list(header_block)

			# ループカウンタを見つける (icmp命令のオペランド)
			loop_counter = None
			for node in nodes:
				node_info = node[0].split()
				if len(node_info) > 1 and 'icmp' in node_info[1]:
					if len(node_info) >= 4 and node_info[3].startswith('%'):
						loop_counter = node_info[3]
						break

			# ループカウンタのソース変数を見つける
			source_var = None
			if loop_counter:
				for node in nodes:
					node_info = node[0].split()
					if len(node_info) > 2 and node_info[2] == loop_counter:
						if 'load' in node_info[1] and len(node_info) > 3:
							source_var = node_info[3]
							break

			if loop_counter and source_var:
				loop_sources[level] = (source_var, loop_counter)
				#print(f"  Loop level {level}: source={source_var}, counter={loop_counter}")

		# 2. インデックスレジスタの定義と使用箇所の探索
		for level, loop_info in self.loop_levels.items():
			sources = loop_sources.get(level, (None, None))

			if not sources[0] or not sources[1]:
				continue

			#print(f"  Checking if {index_reg} is derived from loop level {level}")
			#print(f"    Loop source: {sources[0]}, counter: {sources[1]}")

			# ループの各ブロックを探索
			for block_id in loop_info.nodes:
				nodes = self._read_node_list(block_id)

				# このレジスタの定義または使用を探す
				for node in nodes:
					node_info = node[0].split()

					# 定義を探す
					if len(node_info) > 2 and node_info[2] == index_reg:
						#print(f"    Found definition in block {block_id}: {' '.join(node_info)}")

						# ソースオペランドをチェック
						for i in range(3, len(node_info)):
							operand = node_info[i]

							# 直接のループソースまたはカウンタとの一致
							if operand == sources[0] or operand == sources[1]:
								#print(f"    Direct match: {operand} is loop level {level} variable")
								result[index_reg] = level
								return result

							# 間接的な依存関係を探索
							if operand.startswith('%'):
								# オペランドがループ変数から派生したものか再帰的に確認
								if self._is_derived_from_loop_var(operand, sources[0], sources[1], block_id):
									#print(f"    Indirect match: {index_reg} derived from loop level {level} via {operand}")
									result[index_reg] = level
									return result

		# 3. 追加のヒューリスティック: GEP命令でのアクセスパターンを分析
		if not result:
			#print(f"  No direct derivation found, checking GEP patterns")
			result = self._analyze_gep_patterns(index_reg)

		if not result:
			print(f"  Failed to trace {index_reg} to any loop level")

		return result

	def _is_derived_from_loop_var(self, reg: str, source_var: str, loop_counter: str, block_id: str, visited: Set[str] = None) -> bool:
		"""
		レジスタがループ変数（ソースまたはカウンタ）から派生したものか確認
		"""
		if visited is None:
			visited = set()

		if reg in visited:
			return False

		visited.add(reg)

		# 直接の一致
		if reg == source_var or reg == loop_counter:
			return True

		# このレジスタの定義を探す
		for check_block in self.all_nodes.keys():
			nodes = self._read_node_list(check_block)

			for node in nodes:
				node_info = node[0].split()

				# 定義を見つけた
				if len(node_info) > 2 and node_info[2] == reg:
					# ソースオペランドを確認
					for i in range(3, len(node_info)):
						if i < len(node_info) and node_info[i].startswith('%'):
							# 再帰的に確認
							if self._is_derived_from_loop_var(node_info[i], source_var, loop_counter, check_block, visited):
								return True

		return False

	def _analyze_gep_patterns(self, index_reg: str) -> Dict[str, str]:
		"""
		GEP命令のパターンを分析してループレベルを推定
		"""
		result = {}

		# GEP命令でのインデックスレジスタの使用を探す
		for block_id, nodes in self.all_nodes.items():
			for node in nodes:
				node_info = node[0].split()

				if 'getelementptr' in str(node_info[1]):
					# 配列名を抽出
					array_match = re.search(r'@([a-zA-Z0-9_]+)', str(node_info[1]))
					if not array_match:
						continue

					array_name = array_match.group(1).split('_')[0]

					# この命令でインデックスレジスタが使用されているか確認
					index_positions = []
					for i, part in enumerate(node_info):
						if part == index_reg:
							# インデックスの位置（次元）を特定
							dim_idx = i - 3  # 通常、GEPでは位置-3が次元インデックス
							if dim_idx >= 0:
								index_positions.append(dim_idx)

					if not index_positions:
						continue

					print(f"  Found {index_reg} used in GEP for array {array_name}, dimensions: {index_positions}")

					# 行列積のパターンを認識するヒューリスティック
					# a[i][k], b[k][j], c[i][j] という典型的なパターン
					if array_name == 'a' and 0 in index_positions:
						print(f"  Heuristic: {index_reg} likely from loop level 3 (i) for array {array_name}")
						result[index_reg] = '3'  # i
					elif array_name == 'a' and 1 in index_positions:
						print(f"  Heuristic: {index_reg} likely from loop level 1 (k) for array {array_name}")
						result[index_reg] = '1'  # k
					elif array_name == 'b' and 0 in index_positions:
						print(f"  Heuristic: {index_reg} likely from loop level 1 (k) for array {array_name}")
						result[index_reg] = '1'  # k
					elif array_name == 'b' and 1 in index_positions:
						print(f"  Heuristic: {index_reg} likely from loop level 2 (j) for array {array_name}")
						result[index_reg] = '2'  # j
					elif array_name == 'c' and 0 in index_positions:
						print(f"  Heuristic: {index_reg} likely from loop level 3 (i) for array {array_name}")
						result[index_reg] = '3'  # i
					elif array_name == 'c' and 1 in index_positions:
						print(f"  Heuristic: {index_reg} likely from loop level 2 (j) for array {array_name}")
						result[index_reg] = '2'  # j

		return result

	# レジスタの同一性チェック強化のための新しいヘルパー関数
	def _track_register_equivalence(self, block_id: str, src_reg: str, all_registers: Dict[str, Set[str]]):
		"""
		指定したブロック内で、ソースレジスタから値が流れる先のレジスタを追跡
		"""
		if src_reg not in all_registers:
			all_registers[src_reg] = {src_reg}

		nodes = self._read_node_list(block_id)

		# 追加するレジスタを一時セットに保存
		new_equivalent_regs = set()

		for node in nodes:
			node_info = node[0].split()
			if len(node_info) < 3:
				continue

			# 基本的な値の転送を検出（load命令など）
			if node_info[1].startswith('load') and len(node_info) > 3:
				dest_reg = node_info[2]  # ロード先レジスタ
				addr_reg = node_info[3]  # アドレスレジスタ

				# アドレスレジスタとソースレジスタが関連している場合
				if addr_reg == src_reg or any(addr_reg == r for r in all_registers[src_reg]):
					if dest_reg not in all_registers:
						all_registers[dest_reg] = {dest_reg}

					# 新しい等価レジスタを一時セットに追加
					new_equivalent_regs.add(dest_reg)
					for r in all_registers[src_reg]:
						new_equivalent_regs.add(r)

		# イテレーション後にセットを更新
		for reg in new_equivalent_regs:
			if reg in all_registers:
				all_registers[reg].update(new_equivalent_regs)

		return all_registers

	def _identify_loop_indices_with_details(self) -> Dict[str, Dict[str, Any]]:
		"""各ループレベルのインデックス変数を詳細に特定する"""
		#print("\n=====DEBUG: Identifying loop indices with enhanced method=====")
		loop_indices = {}

		# self.loop_levelsが存在することを確認
		if not hasattr(self, 'loop_levels') or not self.loop_levels:
			print("Warning: loop_levels is not available or empty")
			return loop_indices

		# レジスタ同一性追跡用の辞書
		reg_equivalence = {}  # {reg: set(equivalent_regs)}

		# 各ループレベルを処理
		for level, loop_info in self.loop_levels.items():
			header_block = loop_info.header
			#print(f"\nDEBUG: Analyzing loop level {level}")
			#print(f"  Header: {header_block}")

			# ヘッダーブロックのノードを取得
			header_nodes = self._read_node_list(header_block)

			# ループインデックス情報を初期化
			level_info = {
				'counter': None,     # ループカウンタレジスタ
				'source': None,      # ソースメモリ変数
				'equivalent_regs': set(),  # 同値のレジスタ集合（新規追加）
				'sext_outputs': set(),     # sextの出力レジスタ（新規追加）
				'node_info': {}      # 関連ノード情報
			}

			# 1. icmp命令を探してループカウンタを特定
			for i, node in enumerate(header_nodes):
				node_info = node[0].split()
				if len(node_info) > 1 and 'icmp' in node_info[1]:
					#print(f"  Found icmp in header: {' '.join(node_info)}")

					# icmp命令の詳細な解析
					# 通常、icmp命令は比較演算子、比較値1、比較値2の形式
					# 例: icmp slt i32 %6, 32
					# この場合、%6がループカウンタとなる可能性が高い

					# すべてのオペランドをチェック
					for j in range(3, len(node_info)):
						if node_info[j].startswith('%'):
							# このレジスタを仮のカウンタとして設定
							candidate_counter = node_info[j]

							# このレジスタが実際にループカウンタとして機能するか確認
							# （通常、ループカウンタはループ本体で更新される）
							# まずはload命令からの参照を確認
							for check_node in header_nodes:
								check_info = check_node[0].split()
								if len(check_info) > 3 and 'load' in check_info[1] and check_info[2] == candidate_counter:
									level_info['counter'] = candidate_counter
									level_info['equivalent_regs'].add(candidate_counter)
									level_info['node_info']['icmp'] = {
										'node_id': i,
										'instruction': ' '.join(node_info)
									}
									#print(f"  Loop counter: {level_info['counter']} (verified from load)")
									break

							# カウンタが見つかった場合は終了
							if level_info['counter']:
								break

							# それでも見つからない場合は、単純にこのレジスタをカウンタとして使用
							if not level_info['counter']:
								level_info['counter'] = candidate_counter
								level_info['equivalent_regs'].add(candidate_counter)
								level_info['node_info']['icmp'] = {
									'node_id': i,
									'instruction': ' '.join(node_info)
								}
								#print(f"  Loop counter: {level_info['counter']} (assumed from icmp)")
								break

			# 2. ソースメモリ変数を特定 (load命令から)
			if level_info['counter']:
				for i, node in enumerate(header_nodes):
					node_info = node[0].split()
					# カウンタを定義するload命令を探す
					if (len(node_info) > 2 and node_info[2] == level_info['counter'] and
						'load' in node_info[1]):
						#print(f"  Found load for counter: {' '.join(node_info)}")

						# ロード元のポインタを取得
						if len(node_info) > 3 and node_info[3].startswith('%'):
							level_info['source'] = node_info[3]
							level_info['equivalent_regs'].add(node_info[3])

							level_info['node_info']['load'] = {
								'node_id': i,
								'instruction': ' '.join(node_info)
							}
							#print(f"  Source pointer: {level_info['source']}")
							break

			# 3. レジスタの同値関係を追跡
			if level_info['counter'] or level_info['source']:
				# 同値関係のある全レジスタを追跡
				for reg in list(level_info['equivalent_regs']):
					# ループ内の全ブロックでレジスタの使用を追跡
					for block_id in loop_info.nodes:
						self._track_register_equivalence(block_id, reg, level_info)

				#print(f"  Equivalent registers: {level_info['equivalent_regs']}")

			# 4. sext命令を探す（ループ内の全ブロックで）
			if level_info['counter'] or level_info['source']:
				for block_id in loop_info.nodes:
					nodes = self._read_node_list(block_id)
					for i, node in enumerate(nodes):
						node_info = node[0].split()

						# sext命令をチェック
						if len(node_info) > 3 and 'sext' in node_info[1]:
							source_reg = node_info[3]
							result_reg = node_info[2]

							# このsextがループカウンタまたは同値レジスタから派生しているか
							if source_reg in level_info['equivalent_regs']:
								level_info['sext_outputs'].add(result_reg)
								#print(f"  Found sext: {result_reg} = sext {source_reg}")

								if 'sext_instructions' not in level_info['node_info']:
									level_info['node_info']['sext_instructions'] = []

								level_info['node_info']['sext_instructions'].append({
									'block_id': block_id,
									'node_id': i,
									'instruction': ' '.join(node_info),
									'source': source_reg,
									'result': result_reg
								})

			# 5. GEP命令での使用を探す
			#print(f"  Calling _track_gep_usage for level {level}")
			self._track_gep_usage(level, level_info)

			# 結果をループインデックス情報に追加
			if level_info['counter'] or level_info['source']:
				loop_indices[level] = level_info

		# インデックスレジスタとループレベルのマッピングを構築
		index_reg_to_loop_level = {}
		for level, info in loop_indices.items():
			# カウンタとソースをマッピング
			if info['counter']:
				index_reg_to_loop_level[info['counter']] = level
			if info['source']:
				index_reg_to_loop_level[info['source']] = level

			# 同値レジスタをマッピング
			for reg in info['equivalent_regs']:
				index_reg_to_loop_level[reg] = level

			# sext出力をマッピング
			for reg in info['sext_outputs']:
				index_reg_to_loop_level[reg] = level

		# デバッグ出力
		#print("\nDEBUG: Index register to loop level mapping:")
		#for reg, level in index_reg_to_loop_level.items():
		#	print(f"  {reg} -> Level {level}")

		# マッピング情報を保存
		self.index_reg_to_loop_level = index_reg_to_loop_level

		return loop_indices

	def _track_gep_usage(self, level: str, level_info: Dict) -> None:
		"""
		GEP命令の探索を避け、ポインタレジスタ情報に基づいてループレベルと次元の関連付けを行う
		"""
		#print(f"\nDEBUG: Using pointer register info instead of GEP tracking for loop level {level}")

		# このループレベルに属するブロックを取得
		loop_blocks = []
		if level in self.loop_levels:
			loop_blocks = self.loop_levels[level].nodes

		#print(f"  Loop blocks: {loop_blocks}")

		# 配列次元マッピング情報
		array_dim_mapping = {}  # {array_name: {dimension: level}}

		# ポインタレジスタ情報から配列とインデックスレジスタの関係を取得
		for block_id in loop_blocks:
			if block_id not in self.pointer_regs_info:
				continue

			for reg_info in self.pointer_regs_info[block_id]:
				array_name = reg_info.get('array_name')
				if not array_name or array_name == 'None':
					continue

				# インデックスレジスタのリストを取得
				index_regs_list = reg_info.get('index_regs', {}).get('regs', [])
				if not index_regs_list:
					continue

				#print(f"  Found register info for array {array_name} in block {block_id}")
				#print(f"  Index registers: {index_regs_list}")

				# 各インデックスレジスタがこのループレベルに関連しているか確認
				for i, reg in enumerate(index_regs_list):
					# インデックスレジスタがこのループレベルに属しているか
					is_level_reg = False
					if hasattr(self, 'index_reg_to_loop_level'):
						reg_level = self.index_reg_to_loop_level.get(reg)
						if reg_level == level:
							is_level_reg = True

					if is_level_reg:
						# 次元は0から順に割り当て
						dim_idx = i
						#print(f"  Register {reg} used for dimension {dim_idx} of array {array_name} in level {level}")

						# 配列次元のマッピングを更新
						if array_name not in array_dim_mapping:
							array_dim_mapping[array_name] = {}
						array_dim_mapping[array_name][dim_idx] = level

		# 結果の保存
		if array_dim_mapping:
			level_info['array_dimensions'] = array_dim_mapping
			#print(f"  Array dimensions mapped to this level: {array_dim_mapping}")

			# グローバルなマッピング情報を更新
			if not hasattr(self, 'array_dim_to_loop'):
				self.array_dim_to_loop = {}

			for array_name, dims in array_dim_mapping.items():
				if array_name not in self.array_dim_to_loop:
					self.array_dim_to_loop[array_name] = {}

				# 次元マッピングを更新
				for dim, mapped_level in dims.items():
					self.array_dim_to_loop[array_name][dim] = mapped_level

	def _estimate_dimension_from_position(self, node_info: List[str], position: int) -> int:
		"""
		GEP命令内の位置から次元を推定する一般的な方法
		より正確な次元の推定を実現
		"""
		try:
			# GEP命令の構造を解析
			if 'getelementptr' in node_info[1]:
				# LLVM GEP命令の基本構造:
				# getelementptr [...], [...] @array, i64 0, i64 dim0_idx, i64 dim1_idx, ...

				# 位置がどの次元に対応するかを特定
				array_pos = -1
				array_name = ""

				# まず配列名の位置を特定
				for i, token in enumerate(node_info):
					if token.startswith('@'):
						array_pos = i
						array_name = token[1:].split('_')[0]
						break

				# 配列名が見つからない場合は別の方法で検出
				if array_pos == -1:
					for i, token in enumerate(node_info):
						if 'getelementptr' in token and '@' in token:
							array_match = re.search(r'@([a-zA-Z0-9_]+)', token)
							if array_match:
								array_name = array_match.group(1).split('_')[0]
								array_pos = i
								break

				# 配列名が見つかった場合、次元インデックスの位置を特定
				if array_pos != -1:
					# 最初の次元インデックスは通常、配列名の後の「i64 0」の後に来る
					base_pos = array_pos + 2  # 配列名 + "i64 0" でベース位置を推定

					# 位置が基底インデックスより後ろにあるかどうかを確認
					if position > base_pos:
						# インデックスの位置から次元を計算（0ベース）
						# 通常、「i64 インデックス」のペアで表現されるため、間隔は2
						dim_idx = (position - base_pos - 1) // 2
						return dim_idx

				# 特定の位置パターンが見つからない場合、レジスタの位置に基づいて推定
				# 一般的に、GEP命令内の最初のレジスタが次元0、次が次元1...
				reg_count = 0
				for i in range(1, position):
					if node_info[i].startswith('%'):
						reg_count += 1

				return reg_count

			return 0  # デフォルト値として0（最初の次元）を返す

		except Exception as e:
			print(f"Error estimating dimension: {e}")
			return 0

	def _improve_array_mapping(self):
		"""
		ポインタレジスタ情報とインデックスレジスタのループレベルマッピングを使用して
		配列のアクセスパターンを特定する
		"""
		try:
			#print("\nDEBUG: Improved array dimension mapping using pointer register info")

			# インデックスレジスタとループレベルのマッピングを使用
			index_regs = self.index_reg_to_loop_level
			#print(f"  Using index register mapping: {index_regs}")

			# 配列マッピングの初期化
			array_mapping = {}

			# pointer_regs_infoから配列とインデックスレジスタの関係を取得
			for block_id, regs_list in self.pointer_regs_info.items():
				for reg_info in regs_list:
					array_name = reg_info.get('array_name')
					if not array_name or array_name == 'None':
						continue

					# インデックスレジスタのリストを取得
					index_regs_list = reg_info.get('index_regs', {}).get('regs', [])
					#print(f"  Array {array_name} in block {block_id} uses index registers: {index_regs_list}")

					if not index_regs_list:
						continue

					# 配列がマッピングにない場合は初期化
					if array_name not in array_mapping:
						array_mapping[array_name] = {}

					# 各インデックスレジスタについて、対応するループレベルを確認
					for i, reg in enumerate(index_regs_list):
						if reg in index_regs:
							level = index_regs[reg]
							# 次元は0から順に割り当て
							dim_idx = i

							#print(f"  Found: Array {array_name}, dimension {dim_idx} uses register {reg} (level {level})")

							# マッピング情報を更新
							if dim_idx not in array_mapping[array_name]:
								array_mapping[array_name][dim_idx] = set()
							array_mapping[array_name][dim_idx].add(level)

			# 結果の整理と出力
			final_mapping = {}
			for array_name, dims in array_mapping.items():
				final_mapping[array_name] = {}
				for dim, levels in dims.items():
					# 各次元のレベルを選択（複数ある場合は最初のものを使用）
					levels_list = list(levels)
					if levels_list:
						final_mapping[array_name][dim] = levels_list[0]

			# 結果を出力
			#print("\nFinal array dimension to loop level mapping:")
			#for array_name, dims in final_mapping.items():
			#	print(f"  Array {array_name}:")
			#	for dim, level in dims.items():
			#		print(f"    Dimension {dim} is primarily accessed in loop level: {level}")

			# ループレベル情報にも反映
			for array_name, dims in final_mapping.items():
				for dim, level in dims.items():
					if level in self.loop_levels and array_name not in self.loop_levels[level].array_dims:
						self.loop_levels[level].array_dims[array_name] = {
							'dimensions': self.array_dims.get(array_name, []),
							'accessed_dims': [int(dim)],
							'dim_to_loop': {str(dim): level}
						}
					elif level in self.loop_levels:
						if 'accessed_dims' not in self.loop_levels[level].array_dims.get(array_name, {}):
							self.loop_levels[level].array_dims[array_name]['accessed_dims'] = []
						if int(dim) not in self.loop_levels[level].array_dims[array_name]['accessed_dims']:
							self.loop_levels[level].array_dims[array_name]['accessed_dims'].append(int(dim))
						if 'dim_to_loop' not in self.loop_levels[level].array_dims.get(array_name, {}):
							self.loop_levels[level].array_dims[array_name]['dim_to_loop'] = {}
						self.loop_levels[level].array_dims[array_name]['dim_to_loop'][str(dim)] = level

			return final_mapping

		except Exception as e:
			print(f"Error in _improve_array_mapping: {e}")
			import traceback
			traceback.print_exc()
			return {}

	def _analyze_index_update_pattern_for_level(self, level: str, loop_info: LoopInfo) -> List[RegisterFlow]:
		"""特定のループレベルのインデックス更新パターンを解析"""
		update_patterns = []
		exit_block = loop_info.exit

		# 既存のインデックスレジスタを特定
		level_idx_regs = [reg for reg, lvl in self.index_reg_to_loop_level.items() if lvl == level]

		# 出口ブロックでのインデックス更新命令を検出
		nodes = self._read_node_list(exit_block)
		for node in nodes:
			node_info = node[0].split()
			if len(node_info) < 3:
				continue

			opcode = node_info[1].split('_')[0]
			# 演算命令を特定（add, mul, sub, etc）
			if opcode in ['add', 'mul', 'sub', 'shl', 'or', 'and', 'xor']:
				# このレベルのインデックスレジスタに関連する命令か確認
				is_index_update = False
				for idx_reg in level_idx_regs:
					if any(idx_reg in op for op in node_info[2:]):
						is_index_update = True
						break

				if is_index_update:
					# RegisterFlowオブジェクトを作成
					update_flow = RegisterFlow(
						reg=node_info[2],
						opcode=opcode,
						operands=[op for op in node_info[3:] if self._is_valid_operand(op)],
						output=node_info[2],
						extra_info={'block_id': exit_block, 'instruction': ' '.join(node_info)}
					)
					update_patterns.append(update_flow)

		return update_patterns

	def _analyze_loop_levels(self) -> Dict[str, LoopInfo]:
		result = {}
		try:
			#print("\nDEBUG: Starting _analyze_loop_levels")
			if not self.loops:
				#print("No loops found in the program")
				return result

			# ループ構造の詳細を出力
			#print("\nDEBUG: Loop Structure Details")
			#for idx, loop_nodes in enumerate(self.loops):
			#	# 最外ループがレベル1、最内ループがレベル3という定義に統一
			#	level = str(len(self.loops) - idx)
			#	print(f"Loop {level}: {loop_nodes}")
			#	if len(loop_nodes) >= 2:
			#		print(f"  Header: {loop_nodes[0]}, Exit: {loop_nodes[-1]}")
			#		if len(loop_nodes) > 2:
			#			print(f"  Body nodes: {loop_nodes[1:-1]}")

			# array_dimsの情報を取得
			array_dims = self._get_array_dimensions()
			#print(f"Debug: array_dims = {array_dims}")

			# 各ループレベルのLoopInfoオブジェクトを初期化
			for idx, loop_nodes in enumerate(self.loops):
				level = str(len(self.loops) - idx)  # 最外ループがレベル1、最内ループがレベル3
				#print(f"\nDebug: Processing loop level {level} - {loop_nodes}")

				# 親子関係の設定
				parent_level = str(int(level) - 1) if int(level) > 1 else ""
				children_levels = [str(int(level) + 1)] if int(level) < len(self.loops) else []

				# LoopInfoオブジェクトを作成
				loop_info = LoopInfo(
					nodes=loop_nodes,
					header=loop_nodes[0],
					exit=loop_nodes[-1],
					parent=parent_level,
					children=children_levels,
					array_dims={}
				)
				result[level] = loop_info

			# 作成したループレベル情報をインスタンス変数に代入
			self.loop_levels = result

			# ループインデックス変数を特定
			#print("\nDEBUG: === IDENTIFYING LOOP INDEX VARIABLES ===")
			self._identify_loop_indices_with_details()

			# 各ループレベルのインデックス更新パターンを解析
			for level, loop_info in self.loop_levels.items():
				update_patterns = self._analyze_index_update_pattern_for_level(level, loop_info)
				loop_info.index_updates = update_patterns
				#print(f"  Loop level {level} has {len(update_patterns)} index update patterns")


			# インデックスレジスタとループレベルのマッピングを表示
			#print("\nDEBUG: Index register to loop level mapping:")
			#for reg, level in self.index_reg_to_loop_level.items():
			#	print(f"  {reg} -> Level {level}")

			# 配列の次元とループレベルのマッピングを改善
			#print("\nDEBUG: === IMPROVING ARRAY DIMENSION TO LOOP LEVEL MAPPING ===")
			self.array_dim_to_loop = self._improve_array_mapping()

			return result

		except Exception as e:
			print(f"Error analyzing loop levels: {e}")
			import traceback
			traceback.print_exc()
			return result

	def _analyze_array_access(self, block_id: str, node_info: List[str]) -> Dict[str, Any]:
		# Track registers for each array
		array_accesses = {}
		register_to_array = {}  # Map register -> array_name

		try:
			# 1. First find direct array access (base GEPs)
			for node in node_info:
				node_parts = node[0].split()
				if len(node_parts) < 4:
					continue

				instruction = node_parts[1]
				if 'getelementptr' in instruction and '@' in instruction:
					# Direct array GEP
					array_name = instruction.split('@')[1].split('_')[0]
					result_reg = node_parts[2]
					#print(f"DEBUG: Direct array {array_name} GEP -> {result_reg}")

					if array_name not in array_accesses:
						array_accesses[array_name] = {
							'registers': {
								'gep': [],
								'load': [],
								'store': []
							}
						}
					array_accesses[array_name]['registers']['gep'].append(result_reg)
					register_to_array[result_reg] = array_name

			# 2. Track GEP chains
			for node in node_info:
				node_parts = node[0].split()
				if len(node_parts) < 4:
					continue

				instruction = node_parts[1]
				if 'getelementptr' in instruction:
					result_reg = node_parts[2]
					source_reg = node_parts[-1]
					if source_reg in register_to_array:
						array_name = register_to_array[source_reg]
						array_accesses[array_name]['registers']['gep'].append(result_reg)
						register_to_array[result_reg] = array_name
						#print(f"DEBUG: GEP chain {array_name}: {source_reg} -> {result_reg}")

			# 3. Track load/store using GEP results
			for node in node_info:
				node_parts = node[0].split()
				if len(node_parts) < 4:
					continue

				instruction = node_parts[1].split('_')[0]
				if instruction == 'load':
					pointer = node_parts[3]  # Address to load from
					result = node_parts[2]   # Register receiving value
					if pointer in register_to_array:
						array_name = register_to_array[pointer]
						array_accesses[array_name]['registers']['load'].append(result)
						#print(f"DEBUG: Load from {array_name} using {pointer} -> {result}")

				elif instruction == 'store':
					pointer = node_parts[3]  # Address to store to
					if len(node_parts) > 4:
						value = node_parts[4]    # Value being stored
						if pointer in register_to_array:
							array_name = register_to_array[pointer]
							array_accesses[array_name]['registers']['store'].append(value)
							#print(f"DEBUG: Store to {array_name} using {pointer} <- {value}")

			return array_accesses

		except Exception as e:
			print(f"Error analyzing array access in block {block_id}: {e}")
			return {}

	def _analyze_register_dependency(self, reg: str, nodes: List[List[str]],
								am: List[List[int]], am_size: int) -> Optional[Dict]:
		"""レジスタの依存関係分析"""
		try:
			for line_num, node in enumerate(nodes):
				node = node[0].split()
				if len(node) > 2 and reg in node[1]:
					for src_idx in range(am_size):
						if am[src_idx][line_num]:
							src_node = nodes[src_idx]
							if len(src_node) > 1:
								return {
									'source': src_node[1],
									'type': self._get_dependency_type(src_node[1]),
									'block': str(src_idx)
								}
			return None

		except Exception:
			return None

	def _analyze_pointer_registers(self, block_id: str) -> List[Dict]:
		results = []
		result = {
			'block_id': block_id,
			'gep_node_id': [],
			'array_name': "",
			'term_gep_id': 0,
			'index_regs': {}
		}

		try:
			#print(f"  Analyzing Pointer Register")
			# 1. ノード情報の取得
			nodes = self._read_node_list(block_id)
			if not nodes:
				return results

			# 2. getelementptrノードの収集
			gep_nodes = []
			for line_num, node in enumerate(nodes):
				node = node[0].split()
				if len(node) > 1 and 'getelementptr' in node[1]:
					gep_nodes.append((line_num, node))

			if not gep_nodes:
				result['index_regs'] = {'regs': [], 'array_dim': 0}
				results.append(result)
				return results

			# 3. 終端GEPノードと始端GEPノードの特定
			terminal_geps = self._find_terminal_geps(block_id, gep_nodes)
			begin_geps = self._find_begin_geps(block_id)

			# 4. ループ情報の取得（新しい戻り値形式を使用）
			loop_info = self._analyze_loop_structure(block_id)

			# 5. 各GEPに対する処理
			for terminal_gep in terminal_geps:
				result['gep_node_id'] = [gep[0] for gep in gep_nodes]
				gep_node_id = terminal_gep[0]

				if terminal_gep:
					result['term_gep_id'] = terminal_gep[0]

					# インデックスレジスタの収集
					array_name, reg_info = self._collect_index_registers(block_id, terminal_gep, begin_geps)
					result['array_name'] = array_name
					result['index_regs'] = reg_info

					results.append(result.copy())

			return results

		except Exception as e:
			print(f"Error in analyzing pointer registers for block {block_id}: {e}")
			return results


	def _analyze_loop_structure(self, block_id: str) -> Dict[str, Any]:
		"""
		特定のブロックに関連するループ構造を分析
		Args:
			block_id: 基本ブロックID
		Returns:
			{
				'loop_info': {
					'id': str,            # ループのID
					'level': int,         # ループのネストレベル
					'nodes': List[str],   # ループを構成するノードリスト
					'current': {          # 現在のブロックの情報
						'position': str,  # 'header', 'exit', 'body'のいずれか
						'is_header': bool,
						'is_exit': bool
					}
				},
				'structure': {
					'header': str,      # ヘッダーブロックID
					'exit': str,        # 出口ブロックID
					'body': List[str]   # 本体のブロックIDリスト
				},
				'edges': {
					'forward': List[str],   # 順方向エッジの接続先
					'backward': List[str],  # 逆方向エッジの接続先
					'loop_carried': bool    # ループ伝搬依存の有無
				}
			}
		"""
		try:
			for idx, loop_nodes in enumerate(self.loops):
				if block_id in loop_nodes:
					header = loop_nodes[0]
					exit_node = loop_nodes[-1]

					result = {
						'loop_info': {
							'id': str(idx),
							'level': len(self.loops) - idx,
							'nodes': loop_nodes,
							'current': {
								'position': 'body',
								'is_header': block_id == header,
								'is_exit': block_id == exit_node
							}
						},
						'structure': {
							'header': header,
							'exit': exit_node,
							'body': loop_nodes[1:-1]
						},
						'edges': {
							'forward': [],
							'backward': [],
							'loop_carried': False
						}
					}

					# 位置情報の更新
					if block_id == header:
						result['loop_info']['current']['position'] = 'header'
					elif block_id == exit_node:
						result['loop_info']['current']['position'] = 'exit'

					# エッジ情報の構築
					block_idx = loop_nodes.index(block_id)
					if block_idx < len(loop_nodes) - 1:
						result['edges']['forward'].append(loop_nodes[block_idx + 1])
					if block_idx > 0:
						result['edges']['backward'].append(loop_nodes[block_idx - 1])

					# ループ伝搬依存の確認
					if block_id == header:
						result['edges']['forward'].append(exit_node)
						result['edges']['loop_carried'] = True
					elif block_id == exit_node:
						result['edges']['backward'].append(header)
						result['edges']['loop_carried'] = True

					return result

			# ループに所属しない場合のデフォルト値
			return {
				'loop_info': {
					'id': '',
					'level': 0,
					'nodes': [],
					'current': {
						'position': 'body',
						'is_header': False,
						'is_exit': False
					}
				},
				'structure': {
					'header': '',
					'exit': '',
					'body': []
				},
				'edges': {
					'forward': [],
					'backward': [],
					'loop_carried': False
				}
			}

		except Exception as e:
			print(f"Error analyzing loop structure for block {block_id}: {e}")
			return {
				'loop_info': {'id': '', 'level': 0, 'nodes': [], 'current': {'position': 'body', 'is_header': False, 'is_exit': False}},
				'structure': {'header': '', 'exit': '', 'body': []},
				'edges': {'forward': [], 'backward': [], 'loop_carried': False}
			}

	def _get_terminal_gep(self,
			terminal_geps: List[Tuple[int, List[str]]],
			block_id: str,
			gep_node_id: str) -> Optional[Tuple[int, List[str]]]:
		"""
		指定されたGEPノードに接続する終端GEPノードを取得
		Args:
			terminal_geps: [(line_num, node_info), ...] 形式の終端GEPノードリスト
			block_id: 基本ブロックID
			gep_node_id: 検索対象のGEPノードID
		Returns:
			終端GEPノード (line_num, node_info) または None
		"""
		try:
			# 1. 入力パラメータの検証
			if not terminal_geps or not gep_node_id:
				return None

			# 2. ノードリストの取得
			nodes = self._read_node_list(block_id)
			if not nodes:
				return None

			# 3. AMファイルの読み込み
			#REMOVE
			#am_file = f"{self.r_name}_bblock_{block_id}"
			am_file = f"noundef_bblock_{block_id}"
			am_size, am = AMUtils.Preprocess(self.r_path, am_file)

			# 4. 各終端GEPに対してチェーンの確認
			for term_gep in terminal_geps:
				# GEPチェーンの取得
				gep_chain = self._get_gep_chain(term_gep[0], am, nodes)

				# 指定されたGEPノードがチェーンに含まれるか確認
				for chain_node in gep_chain:
					if int(gep_node_id) == int(chain_node):
						return term_gep

			# 5. 該当するGEPが見つからない場合、最初の終端GEPを返す
			return terminal_geps[0] if terminal_geps else None

		except Exception as e:
			print(f"Error in _get_terminal_gep for block {block_id}: {e}")
			print(f"Context - Terminal GEPs: {len(terminal_geps)}, GEP node ID: {gep_node_id}")
			return None

	def _get_gep_chain(self,
			term_gep_line: int,
			am: List[List[int]],
			nodes: List[List[str]]) -> List[int]:
		"""
		終端GEPからGEPチェーンを取得
		Args:
			term_gep_line: 終端GEPの行番号
			am: 隣接行列
			nodes: ノード情報のリスト
		Returns:
			GEPチェーンを構成するノードの行番号リスト
		"""
		if (term_gep_line, tuple(map(tuple, am))) in self._gep_chain_cache:
			return self._gep_chain_cache[(term_gep_line, tuple(map(tuple, am)))]
		try:
			if term_gep_line >= len(nodes):
				return []

			if 'getelementptr' not in nodes[term_gep_line][0].split()[1]:
				return []

			# GEPチェーンの構築
			gep_chain = [term_gep_line]
			visited = {term_gep_line}
			current_line = term_gep_line

			while True:
				found_prev = False
				# 前方のGEPノードを探索
				for src_idx in range(len(am)):
					if am[src_idx][current_line] and src_idx not in visited:
						if src_idx >= len(nodes):
							continue

						src_node = nodes[src_idx]
						if len(src_node) <= 1:
							continue

						# GEPノードの場合はチェーンに追加
						if 'getelementptr' in src_node[1]:
							gep_chain.append(src_idx)
							visited.add(src_idx)
							current_line = src_idx
							found_prev = True
							break

				if not found_prev:
					break

			self._gep_chain_cache[(term_gep_line, tuple(map(tuple, am)))] = gep_chain
			return gep_chain

		except Exception as e:
			print(f"Error in _get_gep_chain: {e}")
			print(f"Context - Terminal GEP line: {term_gep_line}")
			return []

	def _get_array_from_reg(self, reg: str) -> Optional[str]:
		"""レジスタから対応する配列名を取得"""
		try:
			# ノードからレジスタを利用している配列を探索
			for block_id, nodes in self.all_nodes.items():
				for node in nodes:
					node_info = node[0].split()
					if 'getelementptr' in str(node_info[1]):
						if reg in node_info[2:]:
							# まず現在のノードで@シンボルをチェック
							array_match = re.search(r'@([a-zA-Z0-9_]+)', str(node_info[1]))
							if array_match:
								return array_match.group(1)

							# @シンボルがない場合、AMファイルを使って先行するgetelementptrを探索
							#REMOVE
							#am_file = f"{self.r_name}_bblock_{block_id}"
							am_file = f"noundef_bblock_{block_id}"
							am_size, am = AMUtils.Preprocess(self.r_path, am_file)

							# 現在の行番号を取得
							current_line = int(node_info[0])

							# 先行するノードを探索
							for src_idx in range(am_size):
								if am[src_idx][current_line]:
									if src_idx >= len(nodes):
										continue

									src_node = nodes[src_idx][0].split()
									if 'getelementptr' in src_node[1]:
										array_match = src_node[1].split('_')[1][1:]
										if array_match:
											return array_match
			return None

		except Exception as e:
			print(f"Error in _get_array_from_reg: {e}")
			return None

	def _get_array_dimensions(self) -> Dict[str, List[int]]:
		"""配列の次元情報を収集する"""
		try:
			#print("\nDEBUG: Collecting array dimensions")
			# LLVM IRファイルパス
			llvm_file = os.path.join(self.r_path, f"{self.r_name}.ll")
			#print(f"  Looking for file: {llvm_file}")

			if not os.path.exists(llvm_file):
				print(f"  Warning: LLVM IR file not found: {llvm_file}")
				return {}

			array_dims = {}
			# 配列定義を読み取る
			with open(llvm_file, 'r') as f:
				for line in f:
					#print(f"  Processing line: {line.strip()}")
					if '@' in line and ('global' in line or 'alloca' in line):
						parts = line.split('=')[0].strip()
						array_name = parts.replace('@', '').strip()
						dims: List[int] = []

						# [32 x [32 x i32]] のような形式からサイズを抽出
						parts = line.split('[')
						for part in parts[1:]:
							if 'x' in part:
								size = part.split('x')[0].strip()
								if size.isdigit():
									dims.append(int(size))

						if dims:
							array_dims[array_name] = dims
							#print(f"    Found array {array_name} with dimensions: {dims}")

			#print(f"  Final array dimensions: {array_dims}")
			return array_dims

		except Exception as e:
			print(f"Error in _get_array_dimensions: {e}")
			import traceback
			traceback.print_exc()
			return {}

	def _analyze_control_flow(self) -> Dict:
		"""General-purpose control flow analysis"""
		control_flow = {
			'loops': {},
			'nesting': {},
			'block_order': []
		}

		try:
			# Block connectivity analysis
			block_connectivity = {}
			for block_id in self.all_nodes:
				try:
					am_file = f"noundef_bblock_{block_id}"
					am_size, am = AMUtils.Preprocess(self.r_path, am_file)

					# Initialize block connectivity
					block_connectivity[block_id] = {
						'successors': [],
						'predecessors': []
					}

					# Ensure block_id is valid for matrix size
					curr_block = int(block_id)
					if curr_block < am_size:
						# Collect successors and predecessors
						for i in range(am_size):
							if am[curr_block][i]:
								block_connectivity[block_id]['successors'].append(str(i))
							if am[i][curr_block]:
								block_connectivity[block_id]['predecessors'].append(str(i))
				except Exception as e:
					print(f"Warning: Failed to process block {block_id}: {e}")
					continue

			# Process loop levels
			for level, loop_info in self.loop_levels.items():
				nodes = loop_info.nodes
				header = loop_info.header
				exit = loop_info.exit

				# Get branch info for blocks in this loop
				branches = {}
				for block_id in nodes:
					if block_id in block_connectivity:
						successors = block_connectivity[block_id]['successors']
						if successors:
							branches[block_id] = {
								'targets': successors,
								'type': self._get_branch_type(block_id, successors, header, exit)
							}

				# Add loop level info
				control_flow['loops'][level] = {
					'header': header,
					'exit': exit,
					'body': [n for n in nodes if n not in [header, exit]],
					'branches': branches
				}

				# Add nesting info
				control_flow['nesting'][level] = {
					'parent': loop_info.parent,
					'children': loop_info.children
				}

			# Get block execution order if connectivity exists
			if block_connectivity:
				control_flow['block_order'] = self._get_block_order(block_connectivity)

			return control_flow

		except Exception as e:
			print(f"Error analyzing control flow: {e}")
			return control_flow

	def _get_branch_type(self, block_id: str, targets: List[str], header: str, exit: str) -> str:
		"""General-purpose branch type analysis"""
		if header in targets:
			return 'loop_back'
		elif exit in targets:
			return 'loop_exit'
		elif len(targets) > 1:
			return 'conditional'
		return 'unconditional'

	def _get_block_order(self, connectivity: Dict) -> List[str]:
		"""General-purpose block order analysis"""
		order = []
		visited = set()

		def visit(block: str):
			if block not in visited:
				visited.add(block)
				for succ in connectivity[block]['successors']:
					visit(succ)
				order.append(block)

		entry_blocks = [b for b in connectivity if not connectivity[b]['predecessors']]
		for block in entry_blocks:
			visit(block)

		return list(reversed(order))

	def _analyze_branch_flow(self, block_id: str) -> Dict[str, Any]:
		"""
		分岐命令(br命令)から遡ってデータフローを分析する

		Args:
			block_id: 基本ブロックID

		Returns:
			Dict: {
				'branch_flows': [
					{
						'path_id': str,            # パス識別子
						'condition_node': str,     # 分岐条件ノードID
						'condition_reg': str,      # 分岐条件レジスタ
						'path_nodes': List[str],   # パス上のノードID
						'flow_info': CompFlowInfo, # フロー情報
						'leaf_nodes': List[Dict],  # 葉ノード情報
						'is_taken_branch': bool,   # 条件が真の場合に実行されるパスか
						'execution_path': str      # 実行パス (true/false)
					}
				],
				'branch_deps': {                   # 分岐の依存関係
					'reg_deps': List[str],         # レジスタ依存
					'mem_deps': List[str]          # メモリ依存
				}
			}
		"""
		result = {
			'branch_flows': [],
			'branch_deps': {
				'reg_deps': [],
				'mem_deps': []
			}
		}

		try:
			# 1. branch_leafパスファイルの読み込み
			branch_paths = self._read_path_file(block_id, "branch_leaf")
			if not branch_paths:
				#print(f"No branch-to-leaf paths found for block {block_id}")
				return result

			# 2. ノードリストの取得
			nodes = self._read_node_list(block_id)
			if not nodes:
				print(f"No nodes found for block {block_id}")
				return result

			# 3. パスの解析
			formatted_paths = self._path_formatter(branch_paths)
			if not formatted_paths:
				return result

			#print(f"\nDEBUG: Analyzing branch paths for block {block_id}")
			#print(f"  Found {len(formatted_paths)} path sets")

			# 4. 各パスについて処理
			for path_idx, path_segments in enumerate(formatted_paths):
				#print(f"  Processing path set {path_idx+1} with {len(path_segments)} segments")

				# 分岐ノード情報を収集
				br_node_id = None
				br_node_info = None

				# 最初に分岐ノードを特定
				for node_idx, node in enumerate(nodes):
					node_info = node[0].split()
					if node_info[1].startswith('br'):
						br_node_id = node_idx
						br_node_info = node_info
						#print(f"  Found branch node: {br_node_id}, {' '.join(br_node_info)}")
						break

				# 分岐ノードが見つからない場合はスキップ
				if br_node_id is None:
					#print("  No branch node found in this block")
					continue

				# 各セグメントを処理
				for segment_idx, segment in enumerate(path_segments):
					if not segment:  # 空のセグメントをスキップ
						continue

					#print(f"  Processing segment {segment_idx+1}: {segment}")

					# パスのノード情報を収集
					path_nodes = []
					leaf_nodes = []
					condition_node = None
					condition_reg = None

					# 条件ノード（icmp）と葉ノードを特定
					for node_id in segment:
						node_idx = int(node_id)
						if node_idx >= len(nodes):
							continue

						node_info = nodes[node_idx][0].split()
						path_nodes.append(node_id)

						# 分岐条件（icmp命令）を特定
						if 'icmp' in node_info[1]:
							condition_node = node_id
							condition_reg = node_info[2]  # icmpの結果レジスタ
							#print(f"    Found condition node: {condition_node}, register: {condition_reg}")

						# LEAF ノードの特定
						if 'LEAF' in node_info:
							leaf_nodes.append({
								'node_id': node_id,
								'reg': node_info[1] if len(node_info) > 1 else None
							})
							#print(f"    Found leaf node: {node_id}, register: {node_info[1] if len(node_info) > 1 else None}")

					if condition_node or leaf_nodes:  # 条件ノードまたは葉ノードのいずれかがあれば処理
						# CompFlowInfoの構築
						flows = self._build_branch_flow_info(segment, nodes, block_id)

						# 実行パスの判定（より正確な判定のために分岐情報を解析）
						execution_path = self._determine_execution_path(segment_idx, br_node_info, condition_reg)

						# 真の分岐かどうかの判定
						# segment_idxに基づいて簡易判定（より複雑な判定も可能）
						is_taken_branch = (segment_idx == 0)

						branch_flow = {
							'path_id': f'branch_path_{block_id}_{path_idx}_{segment_idx}',
							'condition_node': condition_node,
							'condition_reg': condition_reg,
							'path_nodes': path_nodes,
							'flow_info': flows,
							'leaf_nodes': leaf_nodes,
							'is_taken_branch': is_taken_branch,
							'execution_path': execution_path
						}

						result['branch_flows'].append(branch_flow)
						#print(f"    Added branch flow with path_id: {branch_flow['path_id']}")

						# 依存関係の収集
						deps = self._collect_branch_dependencies(segment, nodes)
						result['branch_deps']['reg_deps'].extend(deps['reg_deps'])
						result['branch_deps']['mem_deps'].extend(deps['mem_deps'])
					else:
						print(f"    Skipping invalid path: No condition node or leaf nodes found")

			# 重複を排除
			result['branch_deps']['reg_deps'] = list(set(result['branch_deps']['reg_deps']))
			result['branch_deps']['mem_deps'] = list(set(result['branch_deps']['mem_deps']))

			return result

		except Exception as e:
			print(f"Error analyzing branch flow for block {block_id}: {e}")
			import traceback
			traceback.print_exc()
			return result

	def _build_branch_flow_info(self, path_segment: List[str], nodes: List[List[str]], block_id: str) -> CompFlowInfo:
		"""
		パスセグメントからフロー情報を構築

		Args:
			path_segment: パスセグメント（ノードIDのリスト）
			nodes: ブロックのノードリスト
			block_id: 基本ブロックID

		Returns:
			CompFlowInfo: 構築されたフロー情報
		"""
		flow_info = CompFlowInfo(reg_flows=[], block_id=block_id)

		try:
			# 各ノードを処理
			for node_id in path_segment:
				node_idx = int(node_id)
				if node_idx >= len(nodes):
					continue

				node_info = nodes[node_idx][0].split()

				# icmp、load、br、またはその他の重要な命令を処理
				if len(node_info) > 2:
					opcode = node_info[1].split('_')[0]
					if opcode in ['icmp', 'load', 'add', 'mul', 'sub', 'and', 'or', 'xor', 'br']:
						# 命令のオペランドを収集
						operands = []
						for op in node_info[3:]:
							if self._is_valid_operand(op):
								operands.append(op)

						# RegisterFlowオブジェクトを作成
						reg_flow = RegisterFlow(
							reg=node_info[2] if node_info[2].startswith('%') else '',
							opcode=opcode,
							operands=operands,
							output=node_info[2] if node_info[2].startswith('%') else '',
							extra_info={
								'node_id': node_id,
								'instruction': ' '.join(node_info)
							}
						)

						flow_info.add_flow(reg_flow)

			# ループレベルの設定（利用可能な場合）
			for level, info in self.loop_levels.items():
				if block_id in info.nodes:
					flow_info.loop_level = int(level)
					break

		except Exception as e:
			print(f"Error building branch flow info: {e}")

		return flow_info

	def _collect_branch_dependencies(self, path_segment: List[str], nodes: List[List[str]]) -> Dict[str, List[str]]:
		"""
		分岐パスの依存関係を収集

		Args:
			path_segment: パスセグメント（ノードIDのリスト）
			nodes: ブロックのノードリスト

		Returns:
			Dict: {
				'reg_deps': List[str],  # レジスタ依存
				'mem_deps': List[str]   # メモリ依存
			}
		"""
		result = {
			'reg_deps': [],
			'mem_deps': []
		}

		try:
			defined_regs = set()  # このパスで定義されるレジスタ
			used_regs = set()     # このパスで使用されるレジスタ

			# 各ノードを処理
			for node_id in path_segment:
				node_idx = int(node_id)
				if node_idx >= len(nodes):
					continue

				node_info = nodes[node_idx][0].split()

				# レジスタの使用と定義を追跡
				if len(node_info) > 2:
					# 定義されるレジスタ
					if node_info[2].startswith('%'):
						defined_regs.add(node_info[2])

					# 使用されるレジスタ
					for op in node_info[3:]:
						if op.startswith('%'):
							used_regs.add(op)

					# メモリ依存の追跡
					if 'load' in node_info[1] or 'store' in node_info[1]:
						mem_addr = node_info[3] if len(node_info) > 3 else None
						if mem_addr and mem_addr.startswith('%'):
							result['mem_deps'].append(mem_addr)

			# このパスで使用されるが定義されていないレジスタを依存関係として追加
			result['reg_deps'] = list(used_regs - defined_regs)

		except Exception as e:
			print(f"Error collecting branch dependencies: {e}")

		return result

	def _determine_execution_path(self, segment_idx: int, br_node_info: List[str], condition_reg: str) -> str:
		"""
		分岐の実行パスを判定する

		Args:
			segment_idx: パスセグメントのインデックス
			br_node_info: 分岐ノードの情報
			condition_reg: 条件レジスタ

		Returns:
			str: 'true' または 'false'
		"""
		try:
			# 最も単純な判定: segment_idxが0なら真、それ以外なら偽
			if segment_idx == 0:
				return 'true'
			else:
				return 'false'

			# より詳細な実装では、br命令の詳細を解析して判定することも可能
			# 例えば、br命令のオペランドを解析して、true/falseのラベルを特定するなど

		except Exception as e:
			print(f"Error determining execution path: {e}")
			return 'unknown'

	def _is_valid_operand(self, operand: str) -> bool:
		"""
		オペランドが有効か（レジスタか即値か）を判定

		Args:
			operand: チェックするオペランド

		Returns:
			bool: 有効な場合True
		"""
		try:
			# レジスタ
			if operand.startswith('%'):
				return True

			# 即値（整数）
			if operand.isdigit() or (operand.startswith('-') and operand[1:].isdigit()):
				return True

			# 16進数
			if operand.startswith('0x') or operand.startswith('-0x'):
				try:
					int(operand, 16)
					return True
				except ValueError:
					return False

			return False

		except Exception as e:
			print(f"Error checking operand validity: {e}")
			return False

	def analyze(self) -> Dict:
		"""Analyze LLVM IR for AGU and Datapath in general-purpose way"""
		#print("\nDEBUG: Starting analysis")

		# Get array dimensions
		array_dims = self._get_array_dimensions()
		#print(f"DEBUG: Array dimensions: {array_dims}")

		# pointer_registersの解析を追加
		for block_id in self.all_nodes.keys():
			pointer_regs = self._analyze_pointer_registers(block_id)
			if pointer_regs:
				for reg in pointer_regs:
					# ポインタレジスタ情報を蓄積
					self.pointer_regs_info[block_id] = pointer_regs

		#print(f"DEBUG: Pointer registers info: {self.pointer_regs_info}")

		# Get loop levels with array dimensions
		loop_levels = self._analyze_loop_levels()

		# 明示的に配列マッピングを改善
		#print("\nDEBUG: Explicitly calling _improve_array_mapping")
		improved_mapping = self._improve_array_mapping()
		self.array_dim_to_loop = improved_mapping

		# インスタンス変数に設定
		self.array_dim_to_loop = improved_mapping

		# Initialize return structure
		analysis_result = {
			'array_info': {},     # For AGU
			'compute_info': {},   # For Datapath
			'loop_levels': loop_levels,
			'control_flow': self._analyze_control_flow(),
			'branch_flow': {},     # 分岐フロー解析結果を追加
			'array_dim_to_loop': improved_mapping  # 明示的に改善されたマッピングを追加
		}

		# index_reg_to_loop_level マッピングの追加
		if hasattr(self, 'index_reg_to_loop_level'):
			analysis_result['index_reg_to_loop_level'] = self.index_reg_to_loop_level

		# 各ブロックの分岐フロー解析を実行
		for block_id in self.all_nodes.keys():
			branch_flow = self._analyze_branch_flow(block_id)
			if branch_flow and branch_flow.get('branch_flows'):
				analysis_result['branch_flow'][block_id] = branch_flow

		# Collect array access information
		for loop_id, loop_info in loop_levels.items():
			loop_nodes = loop_info.nodes
			for block_id in loop_nodes:
				nodes = self._read_node_list(block_id)
				array_accesses = self._analyze_array_access(block_id, nodes)

				# Update array_info
				for array_name, access_info in array_accesses.items():
					# 配列名の正規化
					base_name = array_name.split('_')[0]
					if base_name not in analysis_result['array_info']:
						analysis_result['array_info'][base_name] = {
							'dimensions': array_dims.get(base_name, []),
							'registers': {'gep': [], 'load': [], 'store': []},
							'loop_access': {}
						}

					array_info = analysis_result['array_info'][base_name]
					# レジスタ情報の更新
					for reg_type in ['gep', 'load', 'store']:
						array_info['registers'][reg_type].extend(
							access_info['registers'][reg_type]
						)
					# ループレベルでのアクセス情報の更新
					if loop_id not in array_info['loop_access']:
						array_info['loop_access'][loop_id] = {
							'gep_regs': access_info['registers']['gep'],
							'load_regs': access_info['registers']['load'],
							'store_regs': access_info['registers']['store']
						}

		# Collect compute information
		compute_paths = self.compute_path_analyzer.analyze_compute_paths()
		analysis_result['compute_info'] = {
			'operations': [],
			'registers': {
				'dependencies': {},
				'flow': {}
			},
			'memory_ops': {
				'loads': [],
				'stores': []
			},
			'branch_conditions': []  # 分岐条件情報を追加
		}

		# 分岐条件情報の追加
		for block_id, branch_info in analysis_result['branch_flow'].items():
			for flow in branch_info.get('branch_flows', []):
				if flow.get('condition_reg'):
					analysis_result['compute_info']['branch_conditions'].append({
						'block_id': block_id,
						'condition_reg': flow['condition_reg'],
						'path_id': flow['path_id'],
						'execution_path': flow.get('execution_path', 'unknown')
					})

		# 変換処理の前に追加
		print("\nDEBUG: Details of compute paths:")
		for i, path in enumerate(compute_paths['compute_paths']):
			print(f"Path {i} (block_id={path.get('computation', {}).get('flow_info', {}).block_id}, type={path.get('type')}):")
			print(f"  Computation sequence: {path.get('computation', {}).get('sequence', [])}")
			print(f"  Inputs: {path.get('inputs', {})}")
			print(f"  Output: {path.get('output', {})}")

		if compute_paths and 'compute_paths' in compute_paths:
			for path in compute_paths['compute_paths']:
				# Add computation sequence
				for comp in path.get('computation', {}).get('sequence', []):
					analysis_result['compute_info']['operations'].append({
						'op': comp['opcode'],
						'output': comp['output_reg'],
						'inputs': comp['input_regs']
					})

				# Add register dependencies
				if 'path_dependencies' in compute_paths:
					for reg_dep in compute_paths['path_dependencies']:
						analysis_result['compute_info']['registers']['dependencies'][
							reg_dep['source_path']] = reg_dep['target_path']

				# Add memory operations
				if 'inputs' in path:
					analysis_result['compute_info']['memory_ops']['loads'].extend(
						path['inputs'].get('loads', [])
					)
				if path.get('output', {}).get('type') == 'memory':
					analysis_result['compute_info']['memory_ops']['stores'].append(
						path['output']
					)

		# Analyzer.analyze()メソッド内の最後の部分に追加
		print("\nDEBUG: Check compute_paths structure:")
		print(f"compute_paths keys: {compute_paths.keys()}")
		if 'compute_paths' in compute_paths:
			print(f"Number of compute paths: {len(compute_paths['compute_paths'])}")
			for i, path in enumerate(compute_paths['compute_paths']):
				print(f"Path {i}: block_id={path.get('computation', {}).get('flow_info', {}).block_id}, type={path.get('type')}")

		#print(f"DEBUG: Analysis completed")
		print(f"analysis_result:{analysis_result}")
		return analysis_result