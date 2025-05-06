///////////////////////////////////////////////////////////////////////////////////////////////////
//
//	ElectronNest
//	Copyright (C) 2024  Shigeyuki TAKANO
//
//  GNU AFFERO GENERAL PUBLIC LICENSE
//	version 3.0
//
//	Load Unit
//		Module Name:	LdUnit
//		Function:
//						Load Unit serves loading data from memory.
//						Top Module of Load Unit
//						Load unit consists of Front-End, Back-End, and Buffer
//
///////////////////////////////////////////////////////////////////////////////////////////////////

module LdUnit
	import	pkg_en::FTk_t;
	import	pkg_en::BTk_t;
	import	pkg_en::WIDTH_DATA;
	import	pkg_extend_index::*;
	import	pkg_mem::DEPTH_FIFO_LD;
#(
	parameter int WIDTH_ADDR		= 8,
	parameter int WIDTH_LENGTH		= 8,
	parameter int EXTERNAL			= 1
)(
	input							clock,
	input							reset,
	input							I_Boot,				//Boot Signal
	input							is_Zero,			//Flag: Shared Data is Zero Value
	input							is_Shared,			//Flag: Block has Shared Data
	input	[WIDTH_LENGTH+1:0]		I_Actual_Length,	//Actual Block-Length
	input	[WIDTH_DATA-1:0]		I_Shared_Data,		//Shared Value
	input	FTk_t					I_FTk,				//Load-Request
	output	BTk_t					O_BTk,				//Backward for Load-Request
	output	FTk_t					O_FTk,				//Send Load-Message
	input	BTk_t					I_BTk,				//Backword for Sending
	input	FTk_t					I_Ld_FTk,			//Loading Data
	output	BTk_t					O_Ld_BTk,			//Backword for Loading
	output	logic					O_Req,				//Load-Request
	output	[1:0]					O_AccessMode,		//Load Access-Mode
	output	[WIDTH_ADDR-1:0]		O_Address,			//Load Address
	output	logic					O_End_Load,			//Flag: End of Loading
	output	logic					O_Busy				//Flag: State in Busy
);

	localparam int WIDTH_FIFO	= $clog2(DEPTH_FIFO_LD);


	//// Connecting Logic											////
	logic	[WIDTH_ADDR+2:0]		ReqLoad;

	logic							We;
	logic							Re;
	logic	[WIDTH_FIFO-1:0]		WAddr;
	logic	[WIDTH_FIFO-1:0]		RAddr;
	logic							Buff_Full;
	logic							Buff_Empty;
	logic	[WIDTH_FIFO:0]			Buff_Num;

	logic							B_Req_BackEnd;
	FTk_t							B_FTk;
	FTk_t							B_Ld_FTk;
	logic	[WIDTH_DATA-1:0]		B_MyID;
	logic	[WIDTH_DATA-1:0]		B_ID_T;
	logic	[WIDTH_DATA-1:0]		B_ID_F;
	logic	[3:0]					B_MetaData;
	logic	[WIDTH_DATA-1:0]		B_AttribWord;
	logic	[WIDTH_DATA-1:0]		B_RConfig;
	logic	[WIDTH_LENGTH-1:0]		B_Length;
	logic	[WIDTH_ADDR-1:0]		B_Stride;
	logic	[WIDTH_ADDR-1:0]		B_Base;

	logic							W_Req_BackEnd;
	FTk_t							W_FTk;
	FTk_t							W_Ld_FTk;
	logic	[WIDTH_DATA-1:0]		W_MyID;
	logic	[WIDTH_DATA-1:0]		W_ID_T;
	logic	[WIDTH_DATA-1:0]		W_ID_F;
	logic	[3:0]					W_MetaData;
	logic	[WIDTH_DATA-1:0]		W_AttribWord;
	logic	[WIDTH_DATA-1:0]		W_RConfig;
	logic	[WIDTH_LENGTH-1:0]		W_Length;
	logic	[WIDTH_ADDR-1:0]		W_Stride;
	logic	[WIDTH_ADDR-1:0]		W_Base;

	logic							W_is_Ack_BackEnd;
	logic							W_is_Matched;
	logic							W_is_Term_Load;
	logic							W_is_Term_BTk;
	logic							W_is_End_Rename;
	logic							W_Req_Stall;

	logic							W_Bypass;
	FTk_t							W_Bypass_FTk;
	BTk_t							W_Bypass_BTk;

	FTk_t							F_Bypass_FTk;
	BTk_t							F_Bypass_BTk;
	FTk_t							B_Bypass_FTk;
	BTk_t							B_Bypass_BTk;

	logic							W_Busy;
	logic							B_Stall;

	logic							SendID;
	logic							F_Empty_Buff;
	logic							F_Full_Buff;

	logic							Read_Buff;

	logic							R_Bypass;
	logic							We_BuffEn;
	logic							Re_BuffEn;


	//// Capturing Logic											////
	logic							Req_BackEnd	[DEPTH_FIFO_LD-1:0];
	FTk_t							FTk			[DEPTH_FIFO_LD-1:0];
	logic	[WIDTH_DATA-1:0]		MyID		[DEPTH_FIFO_LD-1:0];
	logic	[WIDTH_DATA-1:0]		ID_T		[DEPTH_FIFO_LD-1:0];
	logic	[WIDTH_DATA-1:0]		ID_F		[DEPTH_FIFO_LD-1:0];
	logic	[3:0]					MetaData	[DEPTH_FIFO_LD-1:0];
	logic	[WIDTH_DATA-1:0]		AttribWord	[DEPTH_FIFO_LD-1:0];
	logic	[WIDTH_DATA-1:0]		RConfig		[DEPTH_FIFO_LD-1:0];
	logic	[WIDTH_LENGTH-1:0]		Length		[DEPTH_FIFO_LD-1:0];
	logic	[WIDTH_ADDR-1:0]		Stride		[DEPTH_FIFO_LD-1:0];
	logic	[WIDTH_ADDR-1:0]		Base		[DEPTH_FIFO_LD-1:0];


	assign O_Req			= ReqLoad[WIDTH_ADDR+2];
	assign O_AccessMode		= ReqLoad[WIDTH_ADDR+2-1:WIDTH_ADDR];
	assign O_Address		= ReqLoad[WIDTH_ADDR-1:0];

	assign O_End_Load		= W_is_Term_Load;
	assign O_Busy			= W_Busy;

	assign B_Stall			= W_Req_Stall | Buff_Full;
	assign B_Ld_FTk			= W_Ld_FTk;


	//// Buffers													////
	assign We					= B_Req_BackEnd;
	assign Re					= Read_Buff & ~Buff_Empty;

	assign W_FTk				= FTk[ RAddr ];
	always_ff @( posedge clock ) begin
		if ( reset ) begin
			for ( int i=0; i<DEPTH_FIFO_LD; ++i ) begin
				FTk[ i ]		<= '0;
			end
		end
		else if ( We ) begin
			FTk[ WAddr ]		<= B_FTk;
		end
	end

	assign W_Req_BackEnd		= Req_BackEnd[ RAddr ];
	always_ff @( posedge clock ) begin
		if ( reset ) begin
			for ( int i=0; i<DEPTH_FIFO_LD; ++i ) begin
				Req_BackEnd[ i ]<= '0;
			end
		end
		else if ( We | W_Req_BackEnd ) begin
			if ( We ) begin
				Req_BackEnd[ WAddr ]<= B_Req_BackEnd;
			end
			if ( W_Req_BackEnd ) begin
				Req_BackEnd[ RAddr ]<= 1'b0;
			end
		end
	end

	assign W_MyID				= MyID[ RAddr ];
	always_ff @( posedge clock ) begin
		if ( reset ) begin
			for ( int i=0; i<DEPTH_FIFO_LD; ++i ) begin
				MyID[ i ]		<= '0;
			end
		end
		else if ( We ) begin
			MyID[ WAddr ]		<= B_MyID;
		end
	end

	assign W_ID_T				= ID_T[ RAddr ];
	always_ff @( posedge clock ) begin
		if ( reset ) begin
			for ( int i=0; i<DEPTH_FIFO_LD; ++i ) begin
				ID_T[ i ]		<= '0;
			end
		end
		else if ( We ) begin
			ID_T[ WAddr ]		<= B_ID_T;
		end
	end

	assign W_ID_F				= ID_F[ RAddr ];
	always_ff @( posedge clock ) begin
		if ( reset ) begin
			for ( int i=0; i<DEPTH_FIFO_LD; ++i ) begin
				ID_F[ i ]		<= '0;
			end
		end
		else if ( We ) begin
			ID_F[ WAddr ]		<= B_ID_F;
		end
	end

	assign W_MetaData			= MetaData[ RAddr ];
	always_ff @( posedge clock ) begin
		if ( reset ) begin
			for ( int i=0; i<DEPTH_FIFO_LD; ++i ) begin
				MetaData[ i ]	<= '0;
			end
		end
		else if ( We ) begin
			MetaData[ WAddr ]	<= B_MetaData;
		end
	end

	assign W_AttribWord			= AttribWord[ RAddr ];
	always_ff @( posedge clock ) begin
		if ( reset ) begin
			for ( int i=0; i<DEPTH_FIFO_LD; ++i ) begin
				AttribWord[ i ]	<= '0;
			end
		end
		else if ( We ) begin
			AttribWord[ WAddr ]	<= B_AttribWord;
		end
	end

	assign W_RConfig			= RConfig[ RAddr ];
	always_ff @( posedge clock ) begin
		if ( reset ) begin
			for ( int i=0; i<DEPTH_FIFO_LD; ++i ) begin
				RConfig[ i ]	<= '0;
			end
		end
		else if ( We ) begin
			RConfig[ WAddr ]	<= B_RConfig;
		end
	end

	assign W_Length			= Length[ RAddr ];
	always_ff @( posedge clock ) begin
		if ( reset ) begin
			for ( int i=0; i<DEPTH_FIFO_LD; ++i ) begin
				Length[ i ]	<= '0;
			end
		end
		else if ( We ) begin
			Length[ WAddr ]	<= B_Length;
		end
	end

	assign W_Stride			= Stride[ RAddr ];
	always_ff @( posedge clock ) begin
		if ( reset ) begin
			for ( int i=0; i<DEPTH_FIFO_LD; ++i ) begin
				Stride[ i ]	<= '0;
			end
		end
		else if ( We ) begin
			Stride[ WAddr ]	<= B_Stride;
		end
	end

	assign W_Base				= Base[ RAddr ];
	always_ff @( posedge clock ) begin
		if ( reset ) begin
			for ( int i=0; i<DEPTH_FIFO_LD; ++i ) begin
				Base[ i ]	<= '0;
			end
		end
		else if ( We ) begin
			Base[ WAddr ]	<= B_Base;
		end
	end


	Ld_FrontEnd #(
		.WIDTH_ADDR(		WIDTH_ADDR					),
		.WIDTH_LENGTH(		WIDTH_LENGTH				),
		.ExtdConfig(		ExtdConfig					)
	) Ld_FrontEnd
	(
		.clock(				clock						),
		.reset(				reset						),
		.I_Boot(			I_Boot						),
		.I_FTk(				I_FTk						),
		.O_BTk(				O_BTk						),
		.I_Ld_FTk(			B_Ld_FTk					),
		.I_Stall(			B_Stall | F_Full_Buff		),
		.is_Term_Load(		W_is_Term_Load				),
		.is_Ack_BackEnd(	W_is_Ack_BackEnd			),
		.is_Matched(		W_is_Matched				),
		.is_Term_BTk(		W_is_Term_BTk				),
		.is_End_Rename(		W_is_End_Rename				),
		.is_Busy_BackEnd(	W_Busy						),
		.is_Full_Buff(		Buff_Full					),
		.I_Shared(			is_Shared					),
		.I_Actual_Length(	I_Actual_Length				),
		.O_MyID(			B_MyID						),
		.O_ID_T(			B_ID_T						),
		.O_ID_F(			B_ID_F						),
		.O_MetaData(		B_MetaData					),
		.O_FTk(				B_FTk						),
		.O_AttribWord(		B_AttribWord				),
		.O_RConfig(			B_RConfig					),
		.O_Length(			B_Length					),
		.O_Stride(			B_Stride					),
		.O_Base(			B_Base						),
		.O_Bypass(			W_Bypass					),
		.O_Bypass_FTk(		F_Bypass_FTk				),
		.I_Bypass_BTk(		F_Bypass_BTk				),
		.O_SendID(			SendID						),
		.O_Req_BackEnd(		B_Req_BackEnd				)
	);

	RingBuffCTRL #(
		.NUM_ENTRY(			DEPTH_FIFO_LD				)
	) Ld_BuffCTRL
	(
		.clock(				clock						),
		.reset(				reset						),
		.I_We(				We							),
		.I_Re(				Re							),
		.O_WAddr(			WAddr						),
		.O_RAddr(			RAddr						),
		.O_Full(			Buff_Full					),
		.O_Empty(			Buff_Empty					),
		.O_Num(				Buff_Num					)
	);

	assign B_Bypass_FTk		= ( SendID ) ? B_FTk : 			W_Bypass_FTk;
	assign F_Bypass_BTk		= ( SendID ) ? B_Bypass_BTk : 	W_Bypass_BTk;

	always_ff @( posedge clock ) begin
		if ( reset ) begin
			R_Bypass		<= 1'b0;
		end
		else if ( W_Bypass_FTk.v & W_Bypass_FTk.a & W_Bypass_FTk.r ) begin
			R_Bypass		<= 1'b0;
		end
		else if ( W_Bypass ) begin
			R_Bypass		<= 1'b1;
		end
	end

	assign We_BuffEn		= W_Bypass | R_Bypass;
	assign Re_BuffEn		= ~SendID & ~B_Bypass_BTk.n;

	for ( genvar i=0; i<1; ++i) begin
	Buff_IF #(
		.DEPTH_BUFF(		8							),
		.THRESHOLD(			8							),
		.TYPE_FWRD(			FTk_t						)
	) Buff_IF
	(
		.clock(				clock						),
		.reset(				reset						),
		.I_We(				We_BuffEn					),
		.I_Re(				Re_BuffEn					),
		.I_FTk(				F_Bypass_FTk				),
		.O_BTk(				W_Bypass_BTk				),
		.O_FTk(				W_Bypass_FTk				),
		.I_BTk(				B_Bypass_BTk				),
		.O_Empty(			F_Empty_Buff				),
		.O_Full(			F_Full_Buff					)
	);
	end

	Ld_BackEnd #(
		.EXTERNAL(			EXTERNAL					),
		.WIDTH_ADDR(		WIDTH_ADDR					),
		.WIDTH_LENGTH(		WIDTH_LENGTH				)
	) Ld_BackEnd
	(
		.clock(				clock						),
		.reset(				reset						),
		.I_Boot(			I_Boot						),
		.I_Req_BackEnd(		W_Req_BackEnd				),
		.is_Zero(			is_Zero						),
		.is_Shared(			is_Shared					),
		.I_Shared_Data(		I_Shared_Data				),
		.I_MetaData(		W_MetaData  			    ),
		.I_FTk(				W_FTk						),
		.O_FTk(				O_FTk						),
		.I_BTk(				I_BTk						),
		.O_ReqLoad(			ReqLoad						),
		.I_Ld_FTk(			I_Ld_FTk					),
		.O_Ld_BTk(			O_Ld_BTk					),
		.I_MyID(			W_MyID						),
		.I_ID_T(			W_ID_T						),
		.I_ID_F(			W_ID_F						),
		.I_AttribWord(		W_AttribWord				),
		.I_RConfig(			W_RConfig					),
		.I_Length(			W_Length					),
		.I_Stride(			W_Stride					),
		.I_Base(			W_Base						),
		.I_Bypass(			W_Bypass | R_Bypass			),
		.I_Bypass_FTk(		B_Bypass_FTk				),
		.O_Bypass_BTk(		B_Bypass_BTk				),
		.O_Ld_FTk(			W_Ld_FTk					),
		.O_Req_Stall(		W_Req_Stall					),
		.O_is_Term_Load(	W_is_Term_Load				),
		.O_is_Ack_BackEnd(	W_is_Ack_BackEnd			),
		.O_is_Matched(		W_is_Matched				),
		.O_is_Term_BTk(		W_is_Term_BTk				),
		.O_is_End_Rename(	W_is_End_Rename				),
		.O_Read_Buff(		Read_Buff					),
		.O_Busy(			W_Busy						)
	);

endmodule