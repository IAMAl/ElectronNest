///////////////////////////////////////////////////////////////////////////////////////////////////
//
//	ElectronNest
//	Copyright (C) 2024  Shigeyuki TAKANO
//
//  GNU AFFERO GENERAL PUBLIC LICENSE
//	version 3.0
//
//	Simple Buffer Unit
//		Module Name:	Buff_IF
//		Function:
//						Simple Buffer based on Ring Control
//						Buffering Data defined by parameter
//						supports write-enable and read-enable.
//
///////////////////////////////////////////////////////////////////////////////////////////////////

module Buff_IF
	import	pkg_en::FTk_t;
	import	pkg_en::BTk_t;
#(
	parameter int DEPTH_BUFF		= 16,
	parameter int THRESHOLD			= 4,
	parameter type TYPE_FWRD		= FTk_t
)(
	input							clock,
	input							reset,
	input							I_We,				//Write-Enable
	input							I_Re,				//Read-Enable
	input	FTk_t					I_FTk,				//Data
	output	BTk_t					O_BTk,				//Back-Prop Token
	output	FTk_t					O_FTk,				//Data
	input	BTk_t					I_BTk,				//Back-Prop Token
	output	logic					O_Empty,			//Flag: Empty in Buffer
	output	logic					O_Full				//Flag: Full in Buffer
);

	localparam int WIDTH_BUFF		= $clog2(DEPTH_BUFF);

	logic                           Valid;
	logic                           Nack;

	logic							Full;
	logic							Empty;

	logic							Pulse_Nack;
	logic							Pulse_Nack_Bar;

	logic							Nack_Bar1;
	logic							Nack_Bar2;
	logic							Nack_Bar3;

	logic	[WIDTH_BUFF:0]			Num;

	TYPE_FWRD						Buff_FTk;

	logic							Start_Out;
	logic							En_Out;

	logic							We;
	logic							Re;


	logic	[1:0]					R_FSM;

	logic							R_Stop;
	logic							R_Full;

	logic							R_Nack;
	logic							R_NackD1;


	//	 Tokens
	assign Valid			= I_FTk.v;
	assign Nack				= I_BTk.n;

	//	 Contiuous Nack Detections
	assign Nack_Bar1		= ~Nack;
	assign Nack_Bar2		= Nack_Bar1 & ~R_Nack;
	assign Nack_Bar3		= Nack_Bar2 & ~R_NackD1;

	//	 Pulse Token Detection
	assign Pulse_Nack		= ~Nack & R_Nack & ~R_NackD1;
	assign Pulse_Nack_Bar	= Nack & ~R_Nack & R_NackD1;

	//	 Enable to Output
	assign En_Out			= Re;

	//	 State in Full
	assign Full				= Num == (DEPTH_BUFF-1);


	//// Output
	//	Forwarrd Tokens
	assign O_FTk			=  ( En_Out & Empty ) ?	I_FTk :
								( En_Out ) ?		Buff_FTk :
													'0;

	//	 Backward Tokens
	assign O_BTk.n			= R_Full;
	assign O_BTk.t			= I_BTk.t;
	assign O_BTk.v			= I_BTk.v;
	assign O_BTk.c			= I_BTk.c;

	assign O_Empty			= Empty;
	assign O_Full			= Full;


	//// Buffer Control												////
	//	 Write-Enable
	assign We				= ( R_Nack | ~Full ) & Valid & I_We;

	//	 Read-Enable
	assign Re				= ( ~R_Stop & ~Empty ) & I_Re;


	//// Start to Send												////
	always_comb begin
		case ( R_FSM )
			2'h0: begin
				/*assign*/ Start_Out	= Nack_Bar1 & ~Pulse_Nack;
			end
			2'h1: begin
				/*assign*/ Start_Out	= Nack_Bar2;
			end
			2'h2: begin
				/*assign*/ Start_Out	= Nack_Bar3;
			end
			default: begin
				/*assign*/ Start_Out	= 1'b0;
			end
		endcase
	end


	//// Nack Token													////
	//	 Capture Nack Token
	always_ff @( posedge clock ) begin
		if ( reset ) begin
			R_Nack			<= '0;
			R_NackD1		<= '0;
		end
		else begin
			R_Nack			<= Nack;
			R_NackD1		<= R_Nack;
		end
	end


	//// Buffer Full State Detection								////
	//	 Continues Output by DEPTH_FIFO/4 after Full-state
	//	 Used for Sending Nack Token
	//logic						R_Full;
	always_ff @( posedge clock ) begin
		if ( reset ) begin
			R_Full			<= 1'b0;
		end
		else if ( Num == ( DEPTH_BUFF - 3 ) ) begin
			R_Full			<= 1'b1;
		end
		else if ( Num < THRESHOLD ) begin
			R_Full			<= 1'b0;
		end
		else begin
			R_Full			<= R_Full;
		end
	end


	//// Stop to Output Control										////
	//	 Control Nack Token and Buffer Write/Read
	always_ff @( posedge clock ) begin
		if ( reset ) begin
			R_Stop			<= 1'b0;
		end
		else if ( Start_Out ) begin
			R_Stop			<= 1'b0;
		end
		else if ( Nack ) begin
			R_Stop			<= 1'b1;
		end
	end


	//// FSM to Control R_Stop FF									////
	always_ff @( posedge clock ) begin
		if ( reset ) begin
			R_FSM			<= '0;
		end
		else case ( R_FSM )
			2'h0: begin
				if ( Pulse_Nack | Pulse_Nack_Bar ) begin
					R_FSM			<= 2'h1;
				end
				else begin
					R_FSM			<= 2'h0;
				end
			end
			2'h1: begin
				if ( Start_Out ) begin
					R_FSM			<= 2'h0;
				end
				else if ( Pulse_Nack ) begin
					R_FSM			<= 2'h2;
				end
				else begin
					R_FSM			<= 2'h1;
				end
			end
			2'h2: begin
				if ( Start_Out ) begin
					R_FSM			<= 2'h1;
				end
				else begin
					R_FSM			<= 2'h2;
				end
			end
			default: begin
				R_FSM			<= 2'h0;
			end
		endcase
	end


	//// Buffer Body												////
	RingBuff #(
		.DEPTH_BUFF(		DEPTH_BUFF					),
		.TYPE_FWRD(			TYPE_FWRD					)
	) RingBuff
	(
		.clock(				clock						),
		.reset(				reset						),
		.I_We(				We							),
		.I_Re(				Re							),
		.I_FTk(				I_FTk						),
		.O_FTk(				Buff_FTk					),
		.O_Full(										),
		.O_Empty(			Empty						),
		.O_Num(				Num							)
	);

endmodule
