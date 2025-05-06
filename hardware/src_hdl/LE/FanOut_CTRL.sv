///////////////////////////////////////////////////////////////////////////////////////////////////
//
//	ElectronNest
//	Copyright (C) 2024  Shigeyuki TAKANO
//
//  GNU AFFERO GENERAL PUBLIC LICENSE
//	version 3.0
//
//	Router Control
//	Module Name:	FanOut_FIFO
//	Function:
//					FanOut Link Element Controller
//					Control storing in buffer and sending message
//
///////////////////////////////////////////////////////////////////////////////////////////////////

module FanOut_FIFO
	import	pkg_en::*;
	import	pkg_link::*;
	import	pkg_extend_index::*;
#(
	parameter int WIDTH_DATA		= 32,
	parameter int WIDTH_LENGTH		= 8,
	parameter int DEPTH_FIFO		= 16,
	parameter int ExtdConfig		= 0
)(
	input							clock,
	input							reset,
	input	FTk_t					I_FTk,				//Forward Tokens
	output	BTk_t					O_BTk,				//Back-Prop Tokens
	output	FTk_t					O_FTk,				//Forward Tokens
	input	BTk_t					I_BTk,				//Back-prop Tokens
	output	[WIDTH_DATA-1:0]		O_Grt,				//Grant
	output	bit2_t					O_InC				//Condition-Code Path
);

	logic						W_Req;
	logic						W_Ack;

	logic	[WIDTH_DATA-1:0]	F_PATH0;
	logic	[WIDTH_DATA-1:0]	F_PATH1;

	logic						Full_Buff;
	logic						We_F_Buff_ID;
	logic						Re_B_Buff_ID;
	logic						We_F_Buff;
	logic						We_B_Buff;
	logic						Re_B_Buff;

	logic						We_Buff;
	logic						Re_Buff;

	FTk_t						F_FTk;
	BTk_t						F_BTk;
	FTk_t						B_FTk_ID;
	FTk_t						B_FTk;
	BTk_t						B_BTk;

	FTk_t						W_FTk;
	FTk_t						V_FTk;

	bit2_t						B_InC;

	logic						Unit_Length;

	logic						NWe_Buff;
	logic						is_Busy;

	BTk_t						H_BTk;

	assign We_Buff			= ( We_F_Buff & ~Unit_Length ) |
								( We_B_Buff & ~We_F_Buff_ID & ~( I_FTk.v & I_FTk.a & ~I_FTk.r ) & ~Unit_Length & ~NWe_Buff );

	assign Re_Buff			= Re_B_Buff;

	assign W_FTk			= ( Re_B_Buff_ID ) ?	B_FTk_ID :
													B_FTk;

	assign V_FTk			= ( We_F_Buff ) ?		F_FTk :
													I_FTk;

	assign H_BTk			= F_BTk | B_BTk;


	FanOut_FrontEnd FanOut_FrontEnd (
		.clock(				clock						),
		.reset(				reset						),
		.I_FTk(				I_FTk						),
		.O_BTk(				O_BTk						),
		.O_FTk(				F_FTk						),
		.I_BTk(				H_BTk						),
		.O_Req(				W_Req						),
		.I_Ack(				W_Ack						),
		.I_Full_Buff(		Full_Buff					),
		.O_We_BUFF_ID(		We_F_Buff_ID				),
		.O_We_BUFF(			We_F_Buff					),
		.O_PATH0(			F_PATH0						),
		.O_PATH1(			F_PATH1						),
		.O_Unit_Length(		Unit_Length					),
		.is_Busy(			is_Busy						),
		.O_NWe(				NWe_Buff					),
		.I_InC(				B_InC						),
		.O_InC(				O_InC						)
	);

	Buff_IF #(
		.DEPTH_BUFF(		8							),
		.THRESHOLD(			6							),
		.TYPE_FWRD(			FTk_t						)
	) IDs
	(
		.clock(				clock						),
		.reset(				reset						),
		.I_FTk(				F_FTk						),
		.O_BTk(											),
		.O_FTk(				B_FTk_ID					),
		.I_BTk(				B_BTk						),
		.I_We(				We_F_Buff_ID				),
		.I_Re(				Re_B_Buff_ID				),
		.O_Empty(										),
		.O_Full(										)
	);

	Buff_IF #(
		.DEPTH_BUFF(		DEPTH_FIFO					),
		.THRESHOLD(			DEPTH_FIFO/2				),
		.TYPE_FWRD(			FTk_t						)
	) Buff
	(
		.clock(				clock						),
		.reset(				reset						),
		.I_FTk(				V_FTk						),
		.O_BTk(				F_BTk						),
		.O_FTk(				B_FTk						),
		.I_BTk(				B_BTk						),
		.I_We(				We_Buff						),
		.I_Re(				Re_Buff						),
		.O_Empty(										),
		.O_Full(			Full_Buff					)
	);

	FanOut_BackEnd FanOut_BackEnd (
		.clock(				clock						),
		.reset(				reset						),
		.I_FTk(				W_FTk						),
		.O_BTk(				B_BTk						),
		.O_FTk(				O_FTk						),
		.I_BTk(				I_BTk						),
		.I_Req(				W_Req						),
		.O_Ack(				W_Ack						),
		.I_Unit_Length(		Unit_Length					),
		.I_Full_Buff(		Full_Buff					),
		.O_Re_BUFF_ID(		Re_B_Buff_ID				),
		.O_We_BUFF(			We_B_Buff					),
		.O_Re_BUFF(			Re_B_Buff					),
		.I_PATH0(			F_PATH0						),
		.I_PATH1(			F_PATH1						),
		.O_is_Busy(			is_Busy						),
		.O_PATH(			O_Grt						),
		.O_InC(				B_InC						)
	);

endmodule