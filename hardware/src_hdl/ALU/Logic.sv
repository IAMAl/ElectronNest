///////////////////////////////////////////////////////////////////////////////////////////////////
//
//	ElectronNest
//	Copyright (C) 2024  Shigeyuki TAKANO
//
//  GNU AFFERO GENERAL PUBLIC LICENSE
//	version 3.0
//
//	Logic
//	Module Name:	LogicUnit
//	Function:
//					execute logic
//
///////////////////////////////////////////////////////////////////////////////////////////////////

module LogicUnit
	import	pkg_en::*;
	import	pkg_alu::*;
(
	input							I_En,				//Enable to Execute
	input	opcode_al_t				I_Opcode,			//Opcode
	input	cond_t					I_Cond,				//Condition LUT
	input	FTk_t					I_OperandA,			//Source Operand
	input	FTk_t					I_OperandB,			//Source Operand
	output	FTk_t					O_Result,			//Result
	input	BTk_t					I_BTk,				//Backward Tokens
	output	BTk_t					O_BTk				//Backward Tokens
);


	//// Opcode Assignment											////
	logic						SelNot;
	logic	[1:0]				SelOp;
	logic	[1:0]				SelReduc;
	logic						OutCondB;
	logic						OutCondF;

	logic						Valid;

	logic	[WIDTH_DATA-1:0]	SourceA;
	logic	[WIDTH_DATA-1:0]	SourceB;

	logic	[WIDTH_DATA-1:0]	ResultNAND;
	logic	[WIDTH_DATA-1:0]	ResultNOR;
	logic	[WIDTH_DATA-1:0]	ResultXOR;
	logic	[WIDTH_DATA-1:0]	ResultALWAY;
	logic	[WIDTH_DATA-1:0]	ResultLogic;

	logic	[WIDTH_DATA-1:0]	ResultAND_R;
	logic	[WIDTH_DATA-1:0]	ResultOR_R;
	logic	[WIDTH_DATA-1:0]	ResultXOR_R;

	logic	[WIDTH_DATA-1:0]	ResultReduc;
	logic	[WIDTH_DATA-1:0]	ResultAlways;
	logic	[WIDTH_DATA-1:0]	ResultNot;
	logic	[WIDTH_DATA*2-1:0]	Result;

	logic	[WIDTH_COND-1:0]	LUTAddr;
	logic						Cond;


	assign SelOp			= I_Opcode[1:0];
	assign SelNot			= I_Opcode[2];
	assign SelReduc			= I_Opcode[3];

	assign OutCondB			= I_Opcode[4];
	assign OutCondF			= I_Opcode[5];


	//// Sources												    ////
	assign SourceA			= I_OperandA.d;
	assign SourceB			= I_OperandB.d;


	//// First-Stage												////
	//	 Logic
	assign ResultNAND		= ~( SourceA & SourceB );
	assign ResultNOR		= ~( SourceA | SourceB );
	assign ResultXOR		= ( SourceA ^ SourceB );
	assign ResultALWAY		= SourceA;
	assign ResultLogic		= ( SelOp == 2'b00 ) ?		ResultALWAY :
								( SelOp == 2'b01 ) ?	ResultNAND :
								( SelOp == 2'b10 ) ?	ResultNOR :
														ResultXOR;

	//	 Reduction
	assign ResultAND_R		=  0 | &SourceA;
	assign ResultOR_R		=  0 | |SourceA;
	assign ResultXOR_R		=  0 | ^SourceA;

	assign ResultReduc		= ( SelOp == 2'b00 ) ?		'0 :
								( SelOp == 2'b01 ) ?	ResultAND_R :
								( SelOp == 2'b10 ) ?	ResultOR_R :
														ResultXOR_R;


	//// Second-Stage												////
	assign ResultAlways		= ( SelReduc ) ? ResultReduc : ResultLogic;
	assign ResultNot		= ( SelReduc ) ? { ResultReduc[WIDTH_DATA-1:1], ~ResultReduc[0] } : ~ResultLogic;
	assign Result			= ( SelNot ) ?	ResultNot : ResultAlways;

	assign Valid			= I_OperandA.v & I_OperandB.v & I_En;

	assign O_Result.v		= Valid;
	assign O_Result.a		= I_OperandA.a;
	assign O_Result.c		= ( OutCondF ) ? Cond : I_OperandA.c;
	assign O_Result.r		= I_OperandA.r;
	`ifdef EXTEND
	assign O_Result.i		= I_OperandA.i;
	`endif
	assign O_Result.d		= Result;


	//// Condition Code Generation									////
	assign LUTAddr			= { 1'b0, I_OperandA.d != '0, Result == '0 };
	assign Cond				= I_Cond[ LUTAddr ];

	assign O_BTk.n			= I_BTk.n;
	assign O_BTk.t			= I_BTk.t;
	assign O_BTk.v			= ( OutCondB ) ? I_OperandA.c : I_BTk.v;
	assign O_BTk.c			= ( OutCondB ) ? Cond : 		I_BTk.c;

endmodule