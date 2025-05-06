///////////////////////////////////////////////////////////////////////////////////////////////////
//
//	ElectronNest
//	Copyright (C) 2024  Shigeyuki TAKANO
//
//  GNU AFFERO GENERAL PUBLIC LICENSE
//	version 3.0
//
//	Configurable RAM
//	Module Name:	CRAM
//	Function:
//					On-chip Memory
//					CRAM supports load and store with byte, 2-byte, 4-byte access modes.
//
///////////////////////////////////////////////////////////////////////////////////////////////////

module CRAM
	import  pkg_en::*;
	import	pkg_extend_index::*;
#(
	parameter int WIDTH_DATA		= 32,
	parameter int WIDTH_ADDR		= 10,
	parameter int WIDTH_LENGTH		= 10,
	parameter int WIDTH_UNIT		= 8,
	parameter int NUM_MEMUNIT		= 4,
	parameter int SIZE_CRAM			= 256,
	parameter int ExtdConfig		= 0
)(
	input							clock,
	input							reset,
	input							I_Boot,							//Boot Signal
	input	FTk_t					I_FTk,							//Input Forward-Tokens
	output	BTk_t					O_BTk,							//Output Backward-Tokens
	output	FTk_t					O_FTk,							//Output Forward-Tokens
	input	BTk_t					I_BTk							//Input Backward-Tokens
);

	localparam int WIDTH_INDEX	= $clog2(SIZE_CRAM);

	//	 Quad Datum Width
	localparam int POS_B0		= WIDTH_DATA/4;
	localparam int POS_B1		= POS_B0*2;
	localparam int POS_B2		= POS_B0*3;
	localparam int POS_B3		= POS_B0*4;

	//	 NUmber of Bytes in a Data Word
	localparam int NUM_BYTES	= WIDTH_DATA / WIDTH_UNIT;


	//// Memory Cell												////
	logic [WIDTH_UNIT-1:0]		Mem [NUM_MEMUNIT-1:0][SIZE_CRAM-1:0];


	//// Store Path													////
	FTk_t						St_FTk_Path;		// Forward for Store
	BTk_t						St_BTk_Path;		// Back-Prop for Store
	logic						St_Req;				// Store Request
	logic [WIDTH_ADDR-1:0]		St_Addr;			// Store Address
	FTk_t						St_Data;			// Store Data
	logic [1:0]					St_Mode;			// Store Mode

	logic [NUM_MEMUNIT-1:0]		W_St_Req;			// Store Request
	logic [WIDTH_ADDR-1:0]		W_St_Addr;			// Store Address
	data_m_t					W_St_Data;			// Store Data

	logic						is_Byte_St;
	logic						is_Short_St;
	logic						is_Int_St;

	logic						Ld_Req;				// Load Request
	logic [WIDTH_ADDR-1:0]		Ld_Addr;			// Load Address
	FTk_t						Ld_Data;			// Load Data
	logic [1:0]					Ld_Mode;			// Load Mode

	logic [WIDTH_ADDR+1:0]		W_Ld_Addr;			// Load Memory Address
	data_m_t					W_Ld_Data;			// Load Memory Data
	logic	[1:0]				Ld_Addr_Word;

	logic	[3:0]				Sel;
	logic	[3:0]				Sel_B;
	logic	[1:0]				Addr_B;


	//// Capture Signal												////
	logic						R_Ld_Req;
	logic	[1:0]				R_Ld_Addr;
	data_m_t					R_O_Data;			// Load Data


	ReLdStUnit #(
		.WIDTH_DATA(		WIDTH_DATA					),
		.WIDTH_ADDR(		WIDTH_ADDR					),
		.WIDTH_LENGTH(		WIDTH_ADDR					),
		.WIDTH_UNIT(		WIDTH_UNIT					),
		.NUM_MEMUNIT(		NUM_MEMUNIT					),
		.SIZE_CRAM(			SIZE_CRAM					),
		.ExtdConfig(		ExtdConfig					)
	) CRAM_LdStUnit
	(
		.clock(				clock						),
		.reset(				reset						),
		.I_Boot(			I_Boot						),
		.I_FTk(				I_FTk						),
		.O_BTk(				O_BTk						),
		.O_FTk(				O_FTk						),
		.I_BTk(				I_BTk						),
		.O_Ld_Req(			Ld_Req						),
		.O_Ld_Mode(			Ld_Mode						),
		.O_Ld_Address(		Ld_Addr						),
		.I_Ld_Data(			Ld_Data						),
		.O_Ld_BTk(										),
		.O_St_Req(			St_Req						),
		.O_St_Mode(			St_Mode						),
		.O_St_Address(		St_Addr						),
		.O_St_Data(			St_Data						),
		.I_St_BTk(			'0							)
	);


	//// Store Selection											////
	//	 Decode for Addressing Mode
	assign is_Byte_St		= St_Mode == 0;
	assign is_Short_St		= St_Mode == 1;
	assign is_Int_St		= St_Mode == 2;

	//	 Byte-Select
	assign Addr_B			= ( is_Byte_St )	?	St_Addr[1:0] :
								( is_Short_St ) ?	{ 1'b0, St_Addr[1] } :
													2'b00;

	assign Sel_B[1:0]		= ( is_Byte_St ) ?		Sel[1:0] :
								( is_Short_St ) ?	{ Sel[0], Sel[0] } :
													2'b00;
	assign Sel_B[3:2]		= ( is_Byte_St ) ?		Sel[3:2] :
								( is_Short_St )?	{ Sel[1], Sel[1] } :
													2'b00;

	Decoder #(
		.NUM_ENTRY(			NUM_BYTES					),
		.LOG_NUM_ENTRY( 	$clog2(NUM_BYTES)			)
	) St_Sel
	(
		.I_Val(				Addr_B						),
		.O_Grt(				Sel							)
	);

	//	 Store Request (Enable)
	for ( genvar index = 0; index < NUM_MEMUNIT; ++index ) begin: g_st_req
		assign W_St_Req[ index ]	= ( is_Byte_St ) ?		St_Req & Sel_B[ index ] :
										( is_Short_St ) ?	St_Req & Sel_B[ index ] :
										( is_Int_St ) ? 	St_Req :
															'0;
	end


	//// Memory Access Section										////
	//	 Store
	assign W_St_Addr		= (   St_Mode == 2'b10 ) ?	St_Addr :
								( St_Mode == 2'b01 ) ?	{1'h0, St_Addr[WIDTH_ADDR-1:1]} :
								( St_Mode == 2'b00 ) ?	{2'h0, St_Addr[WIDTH_ADDR-1:2]} :
																'0;
	always_ff @( posedge clock ) begin : st_data
		for ( int i = 0; i < NUM_MEMUNIT; ++i ) begin
			if ( W_St_Req[ i ] ) begin
				Mem[ i ][ W_St_Addr[7:0] ]	<= W_St_Data[ i ];
			end
		end
	end

	//	 Load
	assign W_Ld_Addr 		= (   Ld_Mode == 2'b10 ) ?	Ld_Addr :
								( Ld_Mode == 2'b01 ) ?	{1'h0, Ld_Addr[WIDTH_ADDR-1:1]} :
								( Ld_Mode == 2'b00 ) ?	{2'h0, Ld_Addr[WIDTH_ADDR-1:2]} :
																'0;

	assign Ld_Addr_Word	= R_Ld_Addr;
	always_ff @( posedge clock ) begin: ff_ld_addr
		if ( reset ) begin
			R_Ld_Addr		<= 2'h0;
		end
		else begin
			R_Ld_Addr		<= Ld_Addr[1:0];
		end
	end

	always_ff @( posedge clock ) begin
		if ( reset ) begin
			R_Ld_Req	<= 1'b0;
		end
		else begin
			R_Ld_Req	<= Ld_Req;
		end
	end

	always_ff @( posedge clock ) begin : ff_ld_data
		if ( reset ) begin
			for ( int i = 0; i < NUM_MEMUNIT; ++i ) begin
				R_O_Data[ i ]	<= '0;
			end
		end else begin
			for ( int i = 0; i < NUM_MEMUNIT; ++i ) begin
				if ( Ld_Req ) begin
					R_O_Data[ i ]	<= Mem[ i ][ W_Ld_Addr ];
				end
			end
		end
	end


	//// Store-Data Separation										////
	always_comb begin: c_st_path
		case ( St_Mode )
			3: begin
				// Reserved
				W_St_Data[0]	= 8'h00;
				W_St_Data[1]	= 8'h00;
				W_St_Data[2]	= 8'h00;
				W_St_Data[3]	= 8'h00;
			end
			2: begin
				W_St_Data[0]	= St_Data.d[POS_B0-1: 0];
				W_St_Data[1]	= St_Data.d[POS_B1-1:POS_B0];
				W_St_Data[2]	= St_Data.d[POS_B2-1:POS_B1];
				W_St_Data[3]	= St_Data.d[POS_B3-1:POS_B2];
			end
			2'h1: begin
					case ( St_Addr[ 0 ] )
						1'h1: begin
							W_St_Data[0]	= 8'h00;
							W_St_Data[1] 	= 8'h00;
							W_St_Data[2] 	= St_Data.d[POS_B2-1:POS_B1];
							W_St_Data[3] 	= St_Data.d[POS_B3-1:POS_B2];
						end
						1'h0: begin
							W_St_Data[0]	= St_Data.d[POS_B0-1: 0];
							W_St_Data[1]	= St_Data.d[POS_B1-1: POS_B0];
							W_St_Data[2]	= 8'h00;
							W_St_Data[3]	= 8'h00;
						end
						default: begin
							W_St_Data	= '0;
						end
					endcase
			end
			2'h0: begin
				// Quater Word
				case ( St_Addr[1:0] )
					2'h0: begin
						W_St_Data[0]	= St_Data.d[POS_B0-1:0];
						W_St_Data[1]	= 8'h00;
						W_St_Data[2]	= 8'h00;
						W_St_Data[3]	= 8'h00;
					end
					2'h1: begin
						W_St_Data[0]	= 8'h00;
						W_St_Data[1]	= St_Data.d[POS_B0-1:0];
						W_St_Data[2]	= 8'h00;
						W_St_Data[3]	= 8'h00;
					end
					2'h2: begin
						W_St_Data[0]	= 8'h00;
						W_St_Data[1]	= 8'h00;
						W_St_Data[2]	= St_Data.d[POS_B0-1:0];
						W_St_Data[3]	= 8'h00;
					end
					2'h3: begin
						W_St_Data[0]	= 8'h00;
						W_St_Data[1]	= 8'h00;
						W_St_Data[2]	= 8'h00;
						W_St_Data[3]	= St_Data.d[POS_B0-1:0];
					end
					default: begin
						W_St_Data		= '0;
					end
				endcase
			end
		endcase
	end


	//// Byte-Load Selection										////
	assign Ld_Data.v		= R_Ld_Req;
	assign Ld_Data.a		= 1'b0;
	assign Ld_Data.c		= 1'b0;
	assign Ld_Data.r		= 1'b0;
	assign Ld_Data.d		= W_Ld_Data;
	assign Ld_Data.i		= 0;
	always_comb begin: c_ld_path
		case ( Ld_Mode )
			2'h3: begin
				// Reserved
				W_Ld_Data[0]	= 8'h00;
				W_Ld_Data[1]	= 8'h00;
				W_Ld_Data[2]	= 8'h00;
				W_Ld_Data[3]	= 8'h00;
			end
			2'h2: begin
				// Single Word
				W_Ld_Data[0]	= R_O_Data[0];
				W_Ld_Data[1]	= R_O_Data[1];
				W_Ld_Data[2]	= R_O_Data[2];
				W_Ld_Data[3]	= R_O_Data[3];
			end
			2'h1: begin
				// Half Word
				case ( Ld_Addr_Word[0] )
					1'h1: begin
						W_Ld_Data[0]	= R_O_Data[2];
						W_Ld_Data[1]	= R_O_Data[3];
						W_Ld_Data[2]	= 8'h00;
						W_Ld_Data[3]	= 8'h00;
					end
					1'h0: begin
						W_Ld_Data[0]	= R_O_Data[0];
						W_Ld_Data[1]	= R_O_Data[1];
						W_Ld_Data[2]	= 8'h00;
						W_Ld_Data[3]	= 8'h00;
					end
					default: begin
						W_Ld_Data		= '0;
					end
				endcase
			end
			2'h0: begin
				// Quater Word
				case ( Ld_Addr_Word[1:0] )
					2'h0: begin
						W_Ld_Data[0]	= R_O_Data[0];
						W_Ld_Data[1]	= 8'h00;
						W_Ld_Data[2]	= 8'h00;
						W_Ld_Data[3]	= 8'h00;
					end
					2'h1: begin
						W_Ld_Data[0]	= R_O_Data[1];
						W_Ld_Data[1]	= 8'h00;
						W_Ld_Data[2]	= 8'h00;
						W_Ld_Data[3]	= 8'h00;
					end
					2'h2: begin
						W_Ld_Data[0]	= R_O_Data[2];
						W_Ld_Data[1]	= 8'h00;
						W_Ld_Data[2]	= 8'h00;
						W_Ld_Data[3]	= 8'h00;
					end
					2'h3: begin
						W_Ld_Data[0]	= R_O_Data[3];
						W_Ld_Data[1]	= 8'h00;
						W_Ld_Data[2]	= 8'h00;
						W_Ld_Data[3]	= 8'h00;
					end
					default: begin
						W_Ld_Data	= '0;
					end
				endcase
			end
			default: begin
				W_Ld_Data	= '0;
			end
		endcase
	end

endmodule
