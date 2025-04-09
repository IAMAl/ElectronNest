import re
from typing import Dict, List, Optional, Set, Any, Union
from dataclasses import dataclass, field
from functools import reduce
import os
from funcs.Analyzer import RegisterFlow


@dataclass
class LoopInfo:
	nodes: List[str]        # ループCFGノード
	header: str             # ループヘッダー
	exit: str               # 出口ノード
	parent: str             # 親ノードID
	children: List[str]     # 子ノードID
	array_dims: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # {配列名: アクセス情報}

@dataclass
class MemoryAccess:
	"""メモリアクセス情報"""
	type: str				# 'load' or 'store'
	reg: str				# アクセス用レジスタ
	level: int				# アクセスが発生するループレベル
	depends_on: List[str]	# 依存するメモリアクセスのレジスタ
	init_value: str = None	# 初期化値（store時のみ）
	value_reg: str = None	# store時の値レジスタ

@dataclass
class MemoryOp:
	"""Memory operation information (Legacy support)"""
	type: str				# 'load' or 'store'
	reg: str
	base_reg: Optional[str] = None
	value: Optional[str] = None

@dataclass
class DimensionInfo:
	"""次元情報"""
	dim_index: int			# 次元番号（外側から0,1,2...）
	size: int				# 次元サイズ
	level: int				# 対応するループレベル
	stride: int				# メモリアクセスのストライド
	gep_regs: List[str]		# この次元のGEPレジスタ
	index_reg: str = None	# インデックスレジスタ名（オプショナルに変更）

@dataclass
class ArrayAccessPattern:
	"""配列アクセスパターン"""
	array_name: str						# 配列名
	dimensions: List[DimensionInfo]		# 各次元の情報
	memory_accesses: List[MemoryAccess]	# メモリアクセス情報
	base_type: str = "i32"				# 要素の型

@dataclass
class GEPChain:
	"""GEP命令チェーン"""
	base_reg: str			# ベースレジスタ
	indices: List[str]		# 使用するインデックスレジスタ
	result_regs: List[str]	# 生成されるレジスタ
	level: int				# 対応するループレベル

@dataclass
class DimensionAccess:
	level: int				# ループレベル
	gep_regs: List[str]		# GEPレジスタ
	load_regs: List[str]	# ロードレジスタ
	store_regs: List[str]	# ストアレジスタ
	array_size: int			# 配列サイズ

@dataclass
class ArrayDimension:
	size: int
	index_reg: str
	level: int
	access: DimensionAccess

@dataclass
class AGUCode:
	"""AGU generation result"""
	ir_code: List[str]
	dimensions: List[ArrayDimension]
	mem_ops: List[MemoryOp]

	def __getitem__(self, key: str) -> Any:
		if key == 'code':
			return self.ir_code
		raise KeyError(f"Invalid key: {key}")

	def get(self, key: str, default: Any = None) -> Any:
		"""Dictionary-like get method"""
		try:
			return self[key]
		except KeyError:
			return default

class AGUGenerator:
	def __init__(self, analysis_result: Dict, r_file_path: str, r_name: str) -> None:
		self.debug = True
		print("Initializing AGU Generator")
		self.r_path = r_file_path
		self.r_name = r_name

		self.control_flow = analysis_result.get('control_flow', {})
		self.branch_flow = analysis_result.get('branch_flow', {})

		# 配列情報をディープコピーする
		import copy
		self.array_info = copy.deepcopy(analysis_result['array_info'])

		# もし次元情報が空の場合、analysis_resultから直接取得
		for array_name, info in self.array_info.items():
			if 'dimensions' not in info or not info['dimensions']:
				if array_name in analysis_result['array_info'] and 'dimensions' in analysis_result['array_info'][array_name]:
					info['dimensions'] = analysis_result['array_info'][array_name]['dimensions']

		# 残りの初期化コード...
		self.loop_levels = analysis_result['loop_levels']
		print(f"  Arrays: {list(self.array_info.keys())}")
		print(f"  Dimensions: {dict((a, info['dimensions']) for a, info in self.array_info.items())}")
		print(f"  Loop levels: {list(self.loop_levels.keys())}")

		# インデックスレジスタとループレベルのマッピングを保存
		self.index_reg_to_loop_level = {}
		if 'index_reg_to_loop_level' in analysis_result:
			self.index_reg_to_loop_level = analysis_result['index_reg_to_loop_level']
			print(f"  Imported index register to loop level mapping: {len(self.index_reg_to_loop_level)} entries")

		# loop_levelsからの情報も統合
		self._enhance_index_mapping(analysis_result)

		self.array_dims = self._get_array_dimensions()
		self.valid_regs = set()

		# array_dim_to_loop マッピングを取得
		self.array_dim_to_loop = {}
		if 'array_dim_to_loop' in analysis_result:
			self.array_dim_to_loop = analysis_result['array_dim_to_loop']
			print(f"  Imported array dimension to loop level mapping")
			for array_name, dims in self.array_dim_to_loop.items():
				print(f"    Array {array_name}: {dims}")

		self._debug_loop_levels()

		# 初期化済みインデックスの追跡用
		self.initialized_indices = {}

		# 有効なレジスタを追跡
		self.valid_regs = set()
		for array_name, info in self.array_info.items():
			for reg_type in ['gep', 'load', 'store']:
				if 'registers' in info and reg_type in info['registers']:
					self.valid_regs.update(info['registers'][reg_type])

		# ループレベルごとの次元サイズを明示的に保存
		self.level_to_dim_size = {}
		for array_name, dims in self.array_dim_to_loop.items():
			for dim, level in dims.items():
				if array_name in self.array_dims and int(dim) < len(self.array_dims[array_name]):
					size = self.array_dims[array_name][int(dim)]
					if level not in self.level_to_dim_size or size > self.level_to_dim_size[level]:
						self.level_to_dim_size[level] = size

	def _enhance_index_mapping(self, analysis_result: Dict) -> None:
		"""
		loop_levelsから追加のインデックスマッピング情報を抽出

		Args:
			analysis_result: Analyzerからの解析結果
		"""
		try:
			print("\nIndex register to loop level mapping")
			initial_count = len(self.index_reg_to_loop_level)

			# loop_levelsからの情報を統合
			for level, loop_info in analysis_result.get('loop_levels', {}).items():
				for array_name, dims_info in loop_info.array_dims.items():
					# パス情報からインデックスレジスタを収集
					for reg, path_info in dims_info.get('path_info', {}).items():
						if reg not in self.index_reg_to_loop_level:
							self.index_reg_to_loop_level[reg] = level
							print(f"  Added mapping: {reg} -> Level {level} from path_info")

					# dim_to_loopマッピングも確認
					for dim, dim_level in dims_info.get('dim_to_loop', {}).items():
						print(f"  Dimension {dim} of array {array_name} is mapped to level {dim_level}")

			added_count = len(self.index_reg_to_loop_level) - initial_count
			print(f"  Enhanced mapping with {added_count} additional entries")
			print(f"  Final mapping: {self.index_reg_to_loop_level}")

		except Exception as e:
			print(f"Error enhancing index mapping: {e}")

	def _get_array_dimensions(self) -> Dict[str, List[int]]:
		"""
		配列の次元情報を取得

		Returns:
			Dict[str, List[int]]: {array_name: [dim1_size, dim2_size, ...]}
		"""
		dims = {}
		for array_name, info in self.array_info.items():
			if self.debug:
				print(f"\nProcessing array: {array_name}")
				print(f"  Info contents: {info}")

			# 'dimensions'が直接提供されている場合はそれを使用
			if 'dimensions' in info:
				dims[array_name] = info['dimensions']
				continue

			# dimensionsがない場合はループアクセス情報から収集
			sizes = []
			loop_access = info.get('loop_access', {})
			if loop_access:
				# ループレベルをソートして処理
				for level in sorted(loop_access.keys(), key=int, reverse=True):
					# ここではサイズ情報を収集するだけ
					# 実際のサイズは後でAGU生成時に動的に扱う
					sizes.append(None)  # None = サイズは動的に決定

			dims[array_name] = sizes

		dims = {}
		for array_name, info in self.array_info.items():
			# 'dimensions'が直接提供されている場合はそれを使用
			if 'dimensions' in info and info['dimensions']:
				dims[array_name] = info['dimensions']
			else:
				# デフォルト値を使用
				dims[array_name] = [32, 32]  # 明示的なデフォルト値
				print(f"  Using default dimensions [32, 32] for array {array_name}")

		if self.debug:
			print("\n  Final dimensions:")
			for array_name, dimensions in dims.items():
				print(f"    {array_name}: {dimensions}")

		return dims

	def _debug_loop_levels(self):
		"""loop_levelsの構造を表示するデバッグ関数"""
		print("\n=== Debug loop_levels structure ===")
		print(f"Type of self.loop_levels: {type(self.loop_levels)}")
		print(f"Keys in loop_levels: {list(self.loop_levels.keys())}")

		for level, loop_info in self.loop_levels.items():
			print(f"\nLevel: {level}")
			print(f"  Type: {type(loop_info)}")
			if isinstance(loop_info, str):
				print(f"  String value: {loop_info}")
			else:
				print(f"  Attributes/Keys: {dir(loop_info) if hasattr(loop_info, '__dict__') else list(loop_info.keys()) if isinstance(loop_info, dict) else 'N/A'}")
				if hasattr(loop_info, 'nodes'):
					print(f"  nodes: {loop_info.nodes}")
				if hasattr(loop_info, 'header'):
					print(f"  header: {loop_info.header}")
				if hasattr(loop_info, 'exit'):
					print(f"  exit: {loop_info.exit}")
				if hasattr(loop_info, 'array_dims'):
					print(f"  array_dims keys: {list(loop_info.array_dims.keys()) if loop_info.array_dims else 'empty'}")

	def _build_control_flow_graph(self) -> Dict[str, Dict[str, str]]:
		"""
		解析結果の control_flow 情報を使用して制御フローグラフを構築
		"""
		cfg = {}

		try:
			print("\nBuilding control flow graph using analysis results")

			# control_flow情報を取得
			control_flow = getattr(self, 'control_flow', {})
			loops_info = control_flow.get('loops', {})
			nesting_info = control_flow.get('nesting', {})
			block_order = control_flow.get('block_order', [])

			# ブロックの役割マッピングを構築（header, body, exit）
			block_roles = {}
			for level, loop_info in loops_info.items():
				header = loop_info.get('header')
				exit_block = loop_info.get('exit')
				body_blocks = loop_info.get('body', [])

				if header:
					block_roles[header] = ('header', level)
				if exit_block:
					block_roles[exit_block] = ('exit', level)
				for block in body_blocks:
					if block not in block_roles or block_roles[block][0] != 'header':
						block_roles[block] = ('body', level)

			print(f"  Block roles: {block_roles}")

			# 各ループレベルの処理
			for level, loop_info in loops_info.items():
				header = loop_info.get('header')
				exit_block = loop_info.get('exit')
				body_blocks = loop_info.get('body', [])

				# ヘッダーブロックの処理
				if header:
					if header not in cfg:
						cfg[header] = {}

					# ヘッダーからの条件分岐
					if body_blocks:
						cfg[header]['true'] = body_blocks[0]
					else:
						cfg[header]['true'] = exit_block

					# 最外ループの場合は'ret'に遷移、それ以外は exit_block に遷移
					if nesting_info.get(level, {}).get('parent'):
						cfg[header]['false'] = exit_block
					else:
						cfg[header]['false'] = 'ret'

				# ボディブロックの処理
				for i, block in enumerate(body_blocks):
					if block not in cfg:
						cfg[block] = {}

					# 次のブロックを決定
					if i < len(body_blocks) - 1:
						# 次のボディブロックへ
						cfg[block]['next'] = body_blocks[i + 1]
					else:
						# 最後のボディブロックの場合
						# 子ループがあるか確認
						child_levels = nesting_info.get(level, {}).get('children', [])
						if child_levels:
							# 子ループのヘッダーへ
							child_level = child_levels[0]  # 最初の子ループ
							child_header = loops_info.get(child_level, {}).get('header')
							if child_header:
								cfg[block]['next'] = child_header
							else:
								# 子ループヘッダーが見つからない場合
								cfg[block]['next'] = exit_block
						else:
							# 子ループがない場合は出口へ
							cfg[block]['next'] = exit_block

				# 出口ブロックの処理
				if exit_block:
					if exit_block not in cfg:
						cfg[exit_block] = {}

					# 親ループがあるか確認
					parent_level = nesting_info.get(level, {}).get('parent')
					if parent_level:
						# 親ループの情報
						parent_info = loops_info.get(parent_level, {})
						parent_header = parent_info.get('header')
						parent_exit = parent_info.get('exit')
						parent_body = parent_info.get('body', [])

						# 親ループでの位置を特定
						if header in parent_body:
							idx = parent_body.index(header)
							if idx < len(parent_body) - 1:
								# 親ループの次のブロックへ
								cfg[exit_block]['next'] = parent_body[idx + 1]
							else:
								# 親ループの最後のブロックなら親の出口へ
								cfg[exit_block]['next'] = parent_exit
						else:
							# 親ループに属していない場合は自分のヘッダーに戻る
							cfg[exit_block]['next'] = header
					else:
						# 親ループがない場合は自分のヘッダーに戻る
						cfg[exit_block]['next'] = header

			# 最終的なCFGを表示
			print("\nFinal control flow graph:")
			for block_id, transitions in sorted(cfg.items()):
				print(f"  Block {block_id} transitions: {transitions}")

			return cfg

		except Exception as e:
			print(f"Error building control flow graph: {e}")
			import traceback
			traceback.print_exc()
			return {}

	def _get_dimension_size_for_level(self, level: str) -> int:
		"""
		指定されたループレベルに対応する次元サイズを取得

		Args:
			level: ループレベル

		Returns:
			int: 次元サイズ（デフォルトは32）
		"""
		# キャッシュから直接次元サイズを取得
		if level in self.level_to_dim_size:
			dim_size = self.level_to_dim_size[level]
			print(f"  Using cached dimension size {dim_size} for level {level}")
			return dim_size

		# それでも見つからなければ、他の方法で検索
		for array_name, dims in self.array_dim_to_loop.items():
			for dim_str, loop_level in dims.items():
				if loop_level == level and array_name in self.array_dims:
					array_dims = self.array_dims.get(array_name, [])
					if array_dims and int(dim_str) < len(array_dims):
						dim_size = array_dims[int(dim_str)]
						# キャッシュに保存
						self.level_to_dim_size[level] = dim_size
						print(f"  Using dimension size {dim_size} for level {level} from array {array_name}")
						return dim_size

		# デフォルト値
		print(f"  Using default dimension size 32 for level {level}")
		return 32  # デフォルト値

	def _build_array_type(self, array_name: str) -> str:
		"""配列の型文字列を構築（動的次元サイズ対応）"""
		try:
			# 配列の次元情報を取得
			dims = None

			# array_dimsから次元情報を取得
			if array_name in self.array_dims:
				dims = self.array_dims[array_name]

			# array_infoから次元情報を取得（array_dimsに情報がない場合）
			if not dims and array_name in self.array_info:
				dims = self.array_info[array_name].get('dimensions')

			# 次元情報がない場合は、array_dim_to_loopから推測
			if not dims and array_name in self.array_dim_to_loop:
				# この配列の次元数を推測
				dim_mapping = self.array_dim_to_loop[array_name]
				max_dim = max([int(dim) for dim in dim_mapping.keys()]) + 1

				# 各次元のサイズを取得
				dims = []
				for dim in range(max_dim):
					dim_str = str(dim)
					if dim_str in dim_mapping:
						level = dim_mapping[dim_str]
						dim_size = self._get_dimension_size_for_level(level)
						dims.append(dim_size)
					else:
						# マッピングに含まれない次元にはデフォルト値を使用
						dims.append(32)

			# それでもなければデフォルト値を使用
			if not dims:
				dims = [32, 32]  # 2次元配列をデフォルトと仮定

			# 型文字列を内側から外側に構築
			type_str = "i32"  # 基本型

			# 各次元でラップ
			for dim in reversed(dims):
				type_str = f"[{dim} x {type_str}]"

			return type_str

		except Exception as e:
			print(f"Error building array type for {array_name}: {e}")
			return "[32 x [32 x i32]]"  # フォールバック

	def _get_indices_for_access(self, array_name: str, block_id: str, reg: str) -> List[str]:
		"""配列アクセスに使用されるインデックスを取得する汎用的な方法"""
		indices = []
		try:
			print(f"    Array {array_name} access in block {block_id}")

			# array_dim_to_loopマッピングから次元とレベルの対応を取得
			array_mapping = self.array_dim_to_loop.get(array_name, {})
			if not array_mapping:
				print(f"  Warning: No dimension mapping found for array {array_name}")
				# 配列の次元数に基づいてデフォルトインデックスを使用
				dims = self.array_dims.get(array_name, [32, 32])
				for i in range(len(dims)):
					indices.append(f"%i{i+1}")
				return indices

			# 配列の次元数を取得
			dims = self.array_dims.get(array_name, [32, 32])
			n_dims = len(dims)

			# 次元ごとのインデックスをマッピングから取得
			dimension_indices = {}
			for dim_str, level in array_mapping.items():
				dim = int(dim_str)
				dimension_indices[dim] = f"%i{level}"

			# 次元番号順にインデックスを追加
			for dim in range(n_dims):
				if dim in dimension_indices:
					indices.append(dimension_indices[dim])
				else:
					# マッピングにない次元はデフォルトインデックスを使用
					indices.append(f"%i{dim+1}")

			print(f"    Dimensions: {dims}")
			print(f"    Mapping: {array_mapping}")
			print(f"    Generated indices: {indices}")

			return indices

		except Exception as e:
			print(f"Error in _get_indices_for_access: {e}")
			import traceback
			traceback.print_exc()
			# フォールバック: 基本的なインデックスを返す
			return [f"%i1", f"%i2"]

	def _identify_block_role(self, block_id: str) -> str:
		"""ブロックの役割を識別（'header', 'body', 'exit', 'unknown'）"""
		for level, loop_info in self.loop_levels.items():
			if block_id == loop_info.header:
				return 'header'
			if block_id == loop_info.exit:
				return 'exit'
			if block_id in loop_info.nodes:
				return 'body'
		return 'unknown'

	def _block_has_array_access(self, array_name: str, block_id: str, level_access: Dict) -> bool:
		"""ブロックが特定の配列にアクセスするかどうかを確認する汎用的な方法"""
		try:
			# ブロックの役割を識別
			role = self._identify_block_role(block_id)
			print(f"    Checking array access for array {array_name} in block {block_id} (role: {role})")

			# ヘッダーブロックは通常アクセスを含まない
			if role == 'header':
				print(f"    Block {block_id} is a loop header, typically doesn't access arrays")
				return False

			# 出口ブロックも通常アクセスを含まない
			if role == 'exit':
				print(f"    Block {block_id} is a loop exit, typically doesn't access arrays")
				return False

			# このブロックが属するループレベルを特定
			block_levels = []
			for level, loop_info in self.loop_levels.items():
				if block_id in loop_info.nodes:
					block_levels.append(level)

			# 配列次元とループレベルのマッピングを確認
			for level in block_levels:
				# 配列次元がこのレベルにマップされているか確認
				for dim, mapped_level in self.array_dim_to_loop.get(array_name, {}).items():
					if mapped_level == level:
						# このループレベルでの配列アクセス情報を確認
						array_info = self.array_info.get(array_name, {})
						level_info = array_info.get('loop_access', {}).get(level, {})

						if level_info and any(level_info.get(reg_type, []) for reg_type in ['gep_regs', 'load_regs', 'store_regs']):
							print(f"    Block {block_id} in loop level {level} likely accesses array {array_name} dimension {dim}")
							if 'store_regs' in level_info:
								print(f"    Level access info: {level_info}")
								explicit_values = [val for val in level_info.get('store_regs', [])
												if isinstance(val, (int, str)) and not str(val).startswith('%')]
								if explicit_values:
									print(f"    Block {block_id} is first body block with store {explicit_values[0]} operation")
							return True

			# 最内ループのボディブロックは特殊な処理が必要な場合がある
			# 例: 行列乗算の最内ループでは複数の配列にアクセスする
			innermost_level = max(self.loop_levels.keys(), key=int) if self.loop_levels else None
			if innermost_level and innermost_level in block_levels:
				loop_info = self.loop_levels[innermost_level]
				body_blocks = [node for node in loop_info.nodes if node != loop_info.header and node != loop_info.exit]

				if block_id in body_blocks:
					# 最内ループのボディブロックは複数の配列アクセスを含む可能性が高い
					print(f"    Block {block_id} is in innermost loop level {innermost_level}, likely accesses arrays in complex pattern")

					# 配列アクセスのパターンを確認
					for other_array, other_info in self.array_info.items():
						other_loop_access = other_info.get('loop_access', {})
						if innermost_level in other_loop_access and array_name != other_array:
							# 他の配列も同じ最内ループでアクセスされている
							# これは計算パターン（例: 行列乗算）の可能性が高い
							print(f"    Multiple arrays accessed in innermost loop: potential computation pattern")
							return True

			print(f"    No definitive access to array {array_name} detected in block {block_id}")
			return False

		except Exception as e:
			print(f"Error in _block_has_array_access: {e}")
			import traceback
			traceback.print_exc()
			return False

	def _build_dimensions(self, array_name: str, info: Dict) -> List[ArrayDimension]:
		"""AGUCodeのdimensions情報を構築"""
		dimensions = []
		try:
			print(f"\nBuilding dimensions for array {array_name}")

			# 配列次元情報を取得
			dims = []
			if 'dimensions' in info and info['dimensions']:
				dims = info['dimensions']
			if not dims and array_name in self.array_dims:
				dims = self.array_dims[array_name]
			if not dims:
				dims = [32, 32]

			print(f"  Using dimensions: {dims}")

			# ループアクセス情報
			loop_access = info.get('loop_access', {})
			print(f"  Loop access info: {list(loop_access.keys())}")

			# 配列次元とループレベルのマッピングを取得
			array_mapping = self.array_dim_to_loop.get(array_name, {})
			print(f"  Array dimension mapping: {array_mapping}")

			# 各次元に対応するArrayDimensionを構築
			for i in range(len(dims)):
				# この次元のループレベルを決定
				level = None

				# 1. まず配列次元マッピングから取得（最優先）
				if str(i) in array_mapping:
					level = array_mapping[str(i)]
					print(f"  Using mapped level {level} for dimension {i}")
				else:
					# マッピングがない場合は次元番号からループレベルを推測
					for dim_str, lvl in array_mapping.items():
						try:
							if int(dim_str) == i:
								level = lvl
								print(f"  Found mapping for dimension {i}: level {level}")
								break
						except ValueError:
							continue

				# 2. ループアクセス情報を使用（次の優先度）
				if not level and loop_access:
					sorted_levels = sorted(loop_access.keys(), key=int)
					if i < len(sorted_levels):
						level = sorted_levels[i]
					else:
						# インデックスが範囲外の場合、循環させる
						level = sorted_levels[i % len(sorted_levels)]
					print(f"  Using level {level} from loop_access for dimension {i}")

				# 3. デフォルト値（最後の手段）
				if not level:
					level = str(i + 1)
					print(f"  Using default level {level} for dimension {i}")

				# このレベルのアクセス情報
				level_access = loop_access.get(level, {})

				# ArrayDimensionオブジェクトを作成
				dim = ArrayDimension(
					size=dims[i],
					index_reg=f"i{level}",
					level=int(level),
					access=DimensionAccess(
						level=int(level),
						gep_regs=level_access.get('gep_regs', []),
						load_regs=level_access.get('load_regs', []),
						store_regs=level_access.get('store_regs', []),
						array_size=dims[i]
					)
				)
				dimensions.append(dim)

			print(f"  Final dimensions mapping: {[(d.level, i) for i, d in enumerate(dimensions)]}")
			return dimensions

		except Exception as e:
			print(f"Error building dimensions: {e}")
			import traceback
			traceback.print_exc()
			return []

	def _debug_generated_code(self, code: List[str], label: str) -> None:
		"""生成されたコードをデバッグ出力"""
		print(f"\n=== Debug: Generated code for {label} ===")
		for i, line in enumerate(code):
			print(f"{i+1:4d}: {line}")
		print(f"=== End of generated code for {label} ===\n")

	def _determine_block_transition(self, block_id: str) -> Dict[str, str]:
		"""
		特定のブロックからの遷移先を決定する汎用アプローチ

		Args:
			block_id: 基本ブロックのID

		Returns:
			Dict[str, str]: 遷移のマッピング（例: {'true': ブロックID, 'false': ブロックID, 'next': ブロックID}）
		"""
		transitions = {}

		try:
			# ブロックの主要な役割とレベルを決定
			primary_role, primary_level = self._determine_primary_role(block_id)

			if primary_role == 'unknown' or not primary_level:
				print(f"  Warning: Could not determine primary role for block {block_id}")
				return transitions

			print(f"    Block {block_id} primary role: {primary_role} at level {primary_level}")

			loop_info = self.loop_levels.get(primary_level)
			if not loop_info:
				return transitions

			# 役割に基づいて遷移を決定
			if primary_role == 'header':
				# ヘッダーは条件分岐: true -> 最初のボディブロック, false -> 出口またはret
				body_blocks = [node for node in loop_info.nodes if node != loop_info.header and node != loop_info.exit]
				if body_blocks:
					transitions['true'] = body_blocks[0]
				else:
					# ボディブロックがない場合は出口へ直接
					transitions['true'] = loop_info.exit

				# 最外ループか確認
				is_outermost = not loop_info.parent
				if is_outermost:
					# 最外ループのfalse条件分岐はretへ
					transitions['false'] = 'ret'
				else:
					transitions['false'] = loop_info.exit

			elif primary_role == 'body':
				# ボディブロックの場合、次のブロックを決定
				body_blocks = [node for node in loop_info.nodes if node != loop_info.header and node != loop_info.exit]

				# 子ループのヘッダーかどうか確認（レベルが異なる）
				is_child_header = False
				for level, info in self.loop_levels.items():
					if level != primary_level and info.header == block_id:
						is_child_header = True
						child_level = level
						break

				if is_child_header:
					# このブロックは子ループのヘッダーでもある
					# 条件分岐として処理
					child_info = self.loop_levels.get(child_level)
					child_body_blocks = [node for node in child_info.nodes if node != child_info.header and node != child_info.exit]

					if child_body_blocks:
						transitions['true'] = child_body_blocks[0]
					else:
						transitions['true'] = child_info.exit

					transitions['false'] = child_info.exit
				else:
					# 通常のボディブロック
					try:
						idx = body_blocks.index(block_id)
						if idx < len(body_blocks) - 1:
							# 次のボディブロックへ
							transitions['next'] = body_blocks[idx + 1]
						else:
							# 最後のボディブロックの場合は出口へ
							transitions['next'] = loop_info.exit
					except ValueError:
						# ブロックがボディリストにない場合
						transitions['next'] = loop_info.exit

			elif primary_role == 'exit':
				# 出口ブロックの場合、インデックス更新後の次の遷移先を決定
				# 最外ループかチェック
				is_outermost = not loop_info.parent

				if is_outermost:
					# 最外ループの場合はret（終了）へ遷移
					transitions['next'] = 'ret'
				elif loop_info.parent:
					parent_info = self.loop_levels.get(loop_info.parent)
					if parent_info:
						# 親ループの次のブロックを特定
						parent_body = [node for node in parent_info.nodes if node != parent_info.header and node != parent_info.exit]
						try:
							# 親ループ内でのこのループの位置
							idx = parent_body.index(loop_info.header)
							if idx < len(parent_body) - 1:
								# 親ループの次のブロックへ
								transitions['next'] = parent_body[idx + 1]
							else:
								# 親ループの最後のブロックなら親の出口へ
								transitions['next'] = parent_info.exit
						except ValueError:
							# ヘッダーが親ボディにない場合
							transitions['next'] = loop_info.header
					else:
						# 親情報がない場合はヘッダーに戻る
						transitions['next'] = loop_info.header
				else:
					# 親ループがない場合はヘッダーに戻る
					transitions['next'] = loop_info.header

			print(f"    Block {block_id} transitions: {transitions}")
			return transitions

		except Exception as e:
			print(f"Error determining transitions for block {block_id}: {e}")
			import traceback
			traceback.print_exc()
			return transitions

	def _process_header_block(self, block_id, level, loop_info, array_name, info):
		"""
		Process a loop header block with proper index initialization for multidimensional arrays
		using loop structure information from the analyzer

		Args:
			block_id: Header block ID
			level: Loop level
			loop_info: LoopInfo object
			array_name: Current array name
			info: Array info

		Returns:
			List[str]: Generated code lines
		"""
		code = []
		code.append(f"{block_id}:")

		# Initialize child loop indices in header blocks
		# This ensures indices are reset at each iteration of the parent loop

		# Level 1 (outermost) header should initialize level 2 indices
		if level == '1' and loop_info.children:
			# Get the child level (typically level 2)
			child_level = loop_info.children[0]
			code.append(f"  store i32 0, i32* %i{child_level}_ptr, align 4")
			print(f"  Resetting level {child_level} index in level {level} header {block_id}")

		# Level 2 (middle) header should initialize level 3 indices
		elif level == '2' and loop_info.children:
			# Get the child level (typically level 3)
			child_level = loop_info.children[0]
			code.append(f"  store i32 0, i32* %i{child_level}_ptr, align 4")
			print(f"  Resetting level {child_level} index in level {level} header {block_id}")

		# Load current loop index register and check condition
		index_reg = f"i{level}"
		code.append(f"  %{index_reg} = load i32, i32* %{index_reg}_ptr, align 4")

		# Get dimension size for this loop level
		dim_size = self._get_dimension_size_for_level(level)
		code.append(f"  %cond_{index_reg} = icmp slt i32 %{index_reg}, {dim_size}")

		# Determine transitions
		transitions = self._determine_block_transition(block_id)
		true_target = transitions.get('true', loop_info.exit)

		# For outermost loop, false transition is to ret
		false_target = transitions.get('false')
		if not loop_info.parent and not false_target:
			false_target = 'ret'
		elif not false_target:
			false_target = loop_info.exit

		print(f"    Header {block_id} transitions: true -> {true_target}, false -> {false_target}")
		code.append(f"  br i1 %cond_{index_reg}, label %{true_target}, label %{false_target}")
		code.append("")

		return code

	def _process_body_block(self, block_id, level, loop_info, array_name, info):
		"""
		Process a loop body block with proper multidimensional array handling

		Args:
			block_id: Body block ID
			level: Loop level
			loop_info: LoopInfo object
			array_name: Current array name
			info: Array info

		Returns:
			List[str]: Generated code lines
		"""
		code = []
		code.append(f"{block_id}:")

		# Process array access
		level_access = info.get('loop_access', {}).get(level, {})
		if self._block_has_array_access(array_name, block_id, level_access):
			# Get indices
			indices = self._get_indices_for_access(array_name, block_id, "")

			if indices:
				# Extend indices with sext instructions
				sext_regs = []
				for i, idx_reg in enumerate(indices):
					sext_reg = f"%sext_{i}_{array_name}_{block_id}"
					code.append(f"  {sext_reg} = sext i32 {idx_reg} to i64")
					sext_regs.append(sext_reg)

				# Generate GEP instruction
				gep_reg = f"%gep_{array_name}_{block_id}"
				indices_str = "i64 0"
				for sext_reg in sext_regs:
					indices_str += f", i64 {sext_reg}"

				array_type = self._build_array_type(array_name)
				code.append(f"  {gep_reg} = getelementptr inbounds {array_type}, {array_type}* @{array_name}, {indices_str}")

				# Add load/store instructions
				# ロードレジスタの処理
				orig_load_reg = None
				if 'load_regs' in level_access and level_access['load_regs']:
					# オリジナルのレジスタ番号を取得
					orig_load_reg = level_access['load_regs'][0] if level_access['load_regs'] else None
					if orig_load_reg and isinstance(orig_load_reg, str) and orig_load_reg.startswith('%'):
						# オリジナルのレジスタ番号をそのまま使用
						load_reg = f"%load_{array_name}_{block_id}"
					else:
						# オリジナルのレジスタがない場合は動的に生成
						load_reg = f"%load_{array_name}_{block_id}"
					code.append(f"  {load_reg} = load i32, i32* {gep_reg}, align 4")

				# ストアレジスタの処理
				if 'store_regs' in level_access and level_access['store_regs']:
					# オリジナルのストアレジスタ番号を取得
					orig_store_reg = level_access['store_regs'][0] if level_access['store_regs'] else None
					if isinstance(orig_store_reg, str) and orig_store_reg.startswith('%'):
						# オリジナルのレジスタ番号を使用
						store_value_reg = orig_store_reg
						# オリジナルのレジスタが使用されていなければ初期化が必要
						# この場合は上位の処理に任せる
					elif orig_load_reg:  # ロードがある場合は計算結果を使用
						# 計算レジスタ
						calc_reg = f"%calc_{array_name}_{block_id}"
						code.append(f"  {calc_reg} = add i32 {load_reg}, 1")
						store_value_reg = calc_reg
					else:
						# 定数値
						if isinstance(orig_store_reg, (int, str)) and not str(orig_store_reg).startswith('%'):
							store_value = orig_store_reg
						else:
							store_value = "0"  # デフォルト値
						store_value_reg = store_value

					code.append(f"  store i32 {store_value_reg}, i32* {gep_reg}, align 4")

		# Determine transitions
		transitions = self._determine_block_transition(block_id)

		# Check if this block is also a header for a child loop
		is_child_header = False
		for child_level, child_info in self.loop_levels.items():
			if child_info.parent == level and child_info.header == block_id:
				is_child_header = True
				print(f"    Block {block_id} is also header for level {child_level}")
				break

		# Generate appropriate branch instruction based on block role
		if is_child_header:
			# Handle as body block for now, real header processing will happen in child loop
			if 'next' in transitions:
				next_block = transitions['next']
				print(f"    Body {block_id} transition (as child header): next -> {next_block}")
				code.append(f"  br label %{next_block}")
			else:
				print(f"    Body {block_id} fallback transition to exit {loop_info.exit}")
				code.append(f"  br label %{loop_info.exit}")
		elif 'true' in transitions and 'false' in transitions:
			# Conditional branch if needed
			true_target = transitions['true']
			false_target = transitions['false']
			print(f"    Body {block_id} conditional transition: true -> {true_target}, false -> {false_target}")

			# Use current loop's index for condition
			index_reg = f"i{level}"
			dim_size = self._get_dimension_size_for_level(level)
			code.append(f"  %cond_special_{block_id} = icmp slt i32 %{index_reg}, {dim_size}")
			code.append(f"  br i1 %cond_special_{block_id}, label %{true_target}, label %{false_target}")
		elif 'next' in transitions:
			# Unconditional branch (normal body block)
			next_block = transitions['next']
			print(f"    Body {block_id} transition: next -> {next_block}")
			code.append(f"  br label %{next_block}")
		else:
			# Fallback
			print(f"    Body {block_id} fallback transition to exit {loop_info.exit}")
			code.append(f"  br label %{loop_info.exit}")

		code.append("")
		return code

	def _process_exit_block(self, block_id, level, loop_info):
		"""
		Process a loop exit block with improved transition handling

		Args:
			block_id: Exit block ID
			level: Loop level
			loop_info: LoopInfo object

		Returns:
			List[str]: Generated code lines
		"""
		code = []
		code.append(f"{block_id}:")

		# Update index register
		index_reg = f"i{level}"

		# Check for custom update patterns
		update_patterns = loop_info.index_updates if hasattr(loop_info, 'index_updates') else []

		if update_patterns:
			print(f"    Using original update pattern for level {level}")
			for update_flow in update_patterns:
				opcode = update_flow.opcode
				operands = update_flow.operands

				# Build operand list
				valid_operands = []

				# Use index register as first operand
				valid_operands.append(f"%{index_reg}")

				# Find suitable second operand
				for op in operands:
					if op in self.index_reg_to_loop_level and self.index_reg_to_loop_level[op] == level:
						# Skip - same level index already added
						continue
					elif op.startswith('%') and not any(reg == op for reg in self.valid_regs):
						# Replace undefined registers with constant
						if len(valid_operands) < 2:
							valid_operands.append("1")
							break
					else:
						# Valid operand
						if len(valid_operands) < 2:
							valid_operands.append(op)
							break

				# Default to adding 1 if second operand not found
				if len(valid_operands) < 2:
					valid_operands.append("1")

				# Join operands
				operand_str = ", ".join(valid_operands)
				code.append(f"  %{index_reg}_next = {opcode} i32 {operand_str}")
		else:
			# Default increment
			code.append(f"  %{index_reg}_next = add i32 %{index_reg}, 1")

		# Store updated value
		code.append(f"  store i32 %{index_reg}_next, i32* %{index_reg}_ptr, align 4")

		# Determine next transition
		transitions = self._determine_block_transition(block_id)
		next_block = transitions.get('next')

		# For outermost loops or when no next block is found
		if not next_block:
			if not loop_info.parent:
				next_block = 'ret'
			else:
				next_block = loop_info.header

		print(f"    Exit {block_id} transition: next -> {next_block}")
		code.append(f"  br label %{next_block}")
		code.append("")

		return code

	def process_loop_hierarchy(self, level, processed, code, array_name, info):
		"""
		Process the loop hierarchy with improved block ordering and initialization

		Args:
			level: Current loop level
			processed: Set of already processed blocks
			code: Generated code lines
			array_name: Current array name
			info: Array info
		"""
		if level not in self.loop_levels:
			return

		loop_info = self.loop_levels[level]
		print(f"\nProcessing loop level {level}, nodes: {loop_info.nodes}")
		print(f"  Header: {loop_info.header}, Exit: {loop_info.exit}")

		# Process blocks in correct order - header, body blocks, exit
		blocks_to_process = []

		# 1. First add header block
		if loop_info.header not in processed:
			blocks_to_process.append((loop_info.header, 'header'))

		# 2. Add body blocks (excluding header and exit)
		body_blocks = [b for b in loop_info.nodes if b != loop_info.header and b != loop_info.exit]
		for block in body_blocks:
			if block not in processed:
				blocks_to_process.append((block, 'body'))

		# 3. Add exit block
		if loop_info.exit not in processed:
			blocks_to_process.append((loop_info.exit, 'exit'))

		# Process each block
		for block_id, role in blocks_to_process:
			# Skip if another level is supposed to handle this block
			primary_role, primary_level = self._determine_primary_role(block_id)
			if primary_level != level:
				print(f"    Skipping block {block_id} - will be processed at its primary level {primary_level}")
				continue

			print(f"  Generating code for block {block_id} (role: {role})")

			# Generate code based on block role
			if role == 'header':
				block_code = self._process_header_block(block_id, level, loop_info, array_name, info)
			elif role == 'exit':
				block_code = self._process_exit_block(block_id, level, loop_info)
			else:  # 'body'
				block_code = self._process_body_block(block_id, level, loop_info, array_name, info)

			code.extend(block_code)
			processed.add(block_id)

		# Process child loops (after all blocks in this level are processed)
		for child_level, child_info in self.loop_levels.items():
			if child_info.parent == level:
				print(f"  Processing child loop level {child_level}")
				self.process_loop_hierarchy(child_level, processed, code, array_name, info)

	# Add this helper method to determine block roles more accurately
	def _determine_primary_role(self, block_id: str) -> tuple:
		"""
		Determine the primary role of a block in the loop hierarchy.

		Args:
			block_id: Block ID to analyze

		Returns:
			tuple: (role, level) where role is 'header', 'exit', or 'body'
		"""
		roles = {}

		# Find all roles this block has in different loop levels
		for level, loop_info in self.loop_levels.items():
			if block_id == loop_info.header:
				roles[level] = 'header'
			elif block_id == loop_info.exit:
				roles[level] = 'exit'
			elif block_id in loop_info.nodes:
				roles[level] = 'body'

		if not roles:
			return ('unknown', '')

		# Prioritize header > exit > body roles
		header_levels = [level for level, role in roles.items() if role == 'header']
		exit_levels = [level for level, role in roles.items() if role == 'exit']
		body_levels = [level for level, role in roles.items() if role == 'body']

		# Prefer most deeply nested level for the chosen role type
		if header_levels:
			primary_level = max(header_levels, key=int)
			return ('header', primary_level)
		elif exit_levels:
			primary_level = max(exit_levels, key=int)
			return ('exit', primary_level)
		else:
			primary_level = max(body_levels, key=int)
			return ('body', primary_level)

	def _identify_relevant_loop_levels(self, array_name: str, info: Dict) -> Set[str]:
		"""
		Identify which loop levels are relevant for processing a specific array
		based on the array's dimension mappings and access patterns.

		Args:
			array_name: Name of the array being processed
			info: Array information structure

		Returns:
			Set[str]: Set of relevant loop level identifiers
		"""
		relevant_levels = set()

		# 1. Add levels from array dimension to loop mapping
		# This identifies which loop level corresponds to each array dimension
		for dim, level in self.array_dim_to_loop.get(array_name, {}).items():
			relevant_levels.add(level)
			print(f"  Loop level {level} is relevant for array {array_name} dimension {dim}")

		# 2. Add levels from loop access information
		# This identifies which loop levels contain array accesses
		for level in info.get('loop_access', {}).keys():
			if level not in relevant_levels:
				relevant_levels.add(level)
				print(f"  Loop level {level} is relevant for array {array_name} (has access patterns)")

		# 3. Ensure parent loops are included
		# We need to process parent loops even if they don't directly access the array
		additional_levels = set()
		for level in list(relevant_levels):
			# Add all parent levels
			current = level
			while current in self.loop_levels and self.loop_levels[current].parent:
				parent = self.loop_levels[current].parent
				if parent not in relevant_levels and parent not in additional_levels:
					additional_levels.add(parent)
					print(f"  Adding parent loop level {parent} for completeness")
				current = parent

		relevant_levels.update(additional_levels)

		# 4. Ensure all referenced blocks in the control flow are covered
		# by including any loop levels that contain blocks referenced in the control flow
		cfg = self._build_control_flow_graph()
		referenced_blocks = self._collect_all_referenced_blocks(cfg)

		# Find which loop levels contain these blocks
		for block_id in referenced_blocks:
			for level, loop_info in self.loop_levels.items():
				if block_id in loop_info.nodes and level not in relevant_levels:
					relevant_levels.add(level)
					print(f"  Adding loop level {level} because it contains referenced block {block_id}")

		print(f"  Relevant loop levels for array {array_name}: {sorted(relevant_levels, key=int)}")
		return relevant_levels

	def _collect_all_referenced_blocks(self, cfg: Dict[str, Dict[str, str]]) -> Set[str]:
		"""
		Collect all blocks that are referenced in the control flow graph

		Args:
			cfg: Control flow graph mapping

		Returns:
			Set[str]: Set of all block IDs referenced in the CFG
		"""
		referenced_blocks = set()

		# Add all source blocks in the CFG
		for block_id in cfg.keys():
			referenced_blocks.add(block_id)

		# Add all target blocks in the CFG
		for block_id, transitions in cfg.items():
			for transition_type in ['true', 'false', 'next']:
				if transition_type in transitions:
					target = transitions[transition_type]
					if target != 'ret':  # 'ret' is a special pseudo-block
						referenced_blocks.add(target)

		return referenced_blocks

	def _identify_outer_loops(self, relevant_levels: Set[str]) -> List[str]:
		"""
		Identify the outermost loops from the set of relevant loop levels

		Args:
			relevant_levels: Set of relevant loop levels

		Returns:
			List[str]: List of outermost loop level identifiers
		"""
		outer_loops = []

		# A loop is considered outermost if it has no parent,
		# or if its parent is not in the set of relevant levels
		for level in relevant_levels:
			if level in self.loop_levels:
				loop_info = self.loop_levels[level]
				if not loop_info.parent or loop_info.parent not in relevant_levels:
					outer_loops.append(level)
					print(f"  Identified level {level} as an outermost loop")

		return sorted(outer_loops, key=int)

	def _ensure_referenced_blocks_generated(self, processed_blocks: Set[str], cfg: Dict[str, Dict[str, str]], code: List[str]) -> None:
		"""
		Ensure all blocks referenced in the control flow graph are generated

		This method identifies blocks that are referenced in the CFG but haven't been processed yet,
		and generates minimal placeholder code for them to ensure a valid LLVM IR structure.

		Args:
			processed_blocks: Set of already processed block IDs
			cfg: Control flow graph mapping
			code: List of generated code lines to append to
		"""
		try:
			# Collect all referenced blocks from the CFG
			referenced_blocks = set()

			# Include source blocks
			for block_id in cfg.keys():
				referenced_blocks.add(block_id)

			# Include target blocks
			for block_id, transitions in cfg.items():
				for transition_type in ['true', 'false', 'next']:
					if transition_type in transitions:
						target = transitions[transition_type]
						if target != 'ret':  # 'ret' is a special pseudo-block
							referenced_blocks.add(target)

			# Find missing blocks that need to be generated
			missing_blocks = referenced_blocks - processed_blocks

			# Skip 'ret' - it's generated separately
			if 'ret' in missing_blocks:
				missing_blocks.remove('ret')

			if missing_blocks:
				print(f"  Generating {len(missing_blocks)} missing referenced blocks: {sorted(missing_blocks)}")

				# Generate each missing block
				for block_id in sorted(missing_blocks):
					self._generate_placeholder_block(block_id, cfg, code)
					processed_blocks.add(block_id)

		except Exception as e:
			print(f"Error in _ensure_referenced_blocks_generated: {e}")
			import traceback
			traceback.print_exc()

	def _generate_placeholder_block(self, block_id: str, cfg: Dict[str, Dict[str, str]], code: List[str]) -> None:
		"""
		Generate a minimal placeholder block for a missing block

		Args:
			block_id: Block ID to generate
			cfg: Control flow graph mapping
			code: List of generated code lines to append to
		"""
		try:
			# Determine the role of this block in the loop structure
			block_level = None
			block_role = 'unknown'

			for level, loop_info in self.loop_levels.items():
				if block_id == loop_info.header:
					block_level = level
					block_role = 'header'
					break
				elif block_id == loop_info.exit:
					block_level = level
					block_role = 'exit'
					break
				elif block_id in loop_info.nodes:
					if not block_level or int(level) > int(block_level):
						block_level = level
						block_role = 'body'

			print(f"  Generating placeholder for block {block_id} (role: {block_role}, level: {block_level})")

			# Start block
			code.append(f"{block_id}:")

			# Determine appropriate branch instruction
			if block_id in cfg:
				transitions = cfg[block_id]

				# Handle different transition types
				if 'true' in transitions and 'false' in transitions:
					# Conditional branch (typically for header blocks)
					true_target = transitions['true']
					false_target = transitions['false']

					# Generate a dummy condition
					index_reg = f"i{block_level}" if block_level else "i1"
					code.append(f"  %dummy_cond_{block_id} = icmp eq i32 0, 0  ; Always true dummy condition")
					code.append(f"  br i1 %dummy_cond_{block_id}, label %{true_target}, label %{false_target}")
					print(f"    Placeholder conditional branch: true -> {true_target}, false -> {false_target}")

				elif 'next' in transitions:
					# Unconditional branch (for body or exit blocks)
					next_block = transitions['next']
					code.append(f"  br label %{next_block}")
					print(f"    Placeholder unconditional branch: next -> {next_block}")

				else:
					# Fallback - branch to ret
					code.append(f"  br label %ret")
					print(f"    Placeholder fallback branch: next -> ret")
			else:
				# Block not in CFG - fallback to ret
				code.append(f"  br label %ret")
				print(f"    Placeholder fallback branch: next -> ret (not in CFG)")

			code.append("")  # Empty line after block

		except Exception as e:
			print(f"Error generating placeholder for block {block_id}: {e}")
			# Provide a minimal valid placeholder in case of error
			code.append(f"{block_id}:")
			code.append(f"  br label %ret  ; Error fallback")
			code.append("")

	def generate(self) -> Dict[str, AGUCode]:
		"""
		Generate AGU code for all arrays with proper multidimensional array index initialization
		using loop structure information from the analyzer
		"""
		result = {}

		try:
			print("\nGenerating AGU code with improved multidimensional array handling")

			# Display array types
			for array_name in self.array_info.keys():
				array_type = self._build_array_type(array_name)
				print(f"  Array {array_name} type: {array_type}")

			# Process each array
			for array_name, info in self.array_info.items():
				print(f"\nProcessing array {array_name}")

				# Initialize code with headers
				code = []
				code.append("; ModuleID = 'agu_code'")
				code.append(f"target datalayout = \"e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128\"")
				code.append(f"target triple = \"x86_64-pc-linux-gnu\"")

				# Array external definition
				array_type = self._build_array_type(array_name)
				code.append(f"@{array_name} = external global {array_type}, align 16")
				code.append("")

				# Function header
				code.append(f"define void @{array_name}_agu() #0 {{")
				code.append("entry:")

				# Initialize other registers used for array access
				for level in sorted(self.loop_levels.keys(), key=int):
					reg_name = f"i{level}_ptr"
					code.append(f"  %{reg_name} = alloca i32, align 4")
					print(f"  Allocated index register for level {level}: %{reg_name}_ptr")

				outermost_level = min(self.loop_levels.keys(), key=int)
				code.append(f"  store i32 0, i32* %i{outermost_level}_ptr, align 4")
				print(f"  Initialized outermost index i{outermost_level} in entry block")

				# Branch to outermost loop header
				relevant_levels = self._identify_relevant_loop_levels(array_name, info)
				if relevant_levels:
					outer_level = min(relevant_levels, key=int)
					outer_header = self.loop_levels[outer_level].header
					code.append(f"  br label %{outer_header}")
					code.append("")
				else:
					# No loops, branch directly to ret
					code.append(f"  br label %ret")
					code.append("")

				# Track processed blocks
				processed_blocks = {"entry"}

				# Process loop hierarchy starting from outermost loop
				outer_levels = self._identify_outer_loops(relevant_levels)
				for level in sorted(outer_levels, key=int):
					self.process_loop_hierarchy(level, processed_blocks, code, array_name, info)

				# Add any missing referenced blocks
				self._ensure_referenced_blocks_generated(processed_blocks,
														self._build_control_flow_graph(),
														code)

				# Add ret block if not already processed
				if "ret" not in processed_blocks:
					code.append("ret:")
					code.append("  ret void")
					processed_blocks.add("ret")

				# Function epilogue
				code.extend([
					"}",
					"",
					"attributes #0 = { nounwind }"
				])

				# Build final dimensions and memory operations
				dimensions = self._build_dimensions(array_name, info)
				mem_ops = []

				# Debug output
				self._debug_generated_code(code, f"{array_name}_final")

				# Create AGUCode object
				result[array_name] = AGUCode(ir_code=code, dimensions=dimensions, mem_ops=mem_ops)

			print("AGU Code generation completed successfully")
			return result

		except Exception as e:
			print(f"Error during AGU generation: {e}")
			import traceback
			traceback.print_exc()
			return result