///////////////////////////////////////////////////////////////////////////////////////////////////
//
//	ElectronNest
//	Copyright (C) 2024  Shigeyuki TAKANO
//
//  GNU AFFERO GENERAL PUBLIC LICENSE
//	version 3.
//
//	Integer Adder and Logic
//	Module Name:	MultLogic
//	Function:
//					Integer Adder and Logic Cluster
//					Integer Adder supports addition and subtraction with signed and unsigned type,
//						and also supports a saturation when a carry-out is there.
//					Logic suports basic four logic operations, its not output, and reductions.
//
///////////////////////////////////////////////////////////////////////////////////////////////////

module AddLogic
import	pkg_en::FTk_t;
import	pkg_en::BTk_t;
import	pkg_alu::*;
(
	input							clock,
	input							reset,
	input 							I_Active,			//Activate Module
	input	opcode_al_t				I_Opcode,			//Opcode
	input	cond_t					I_Cond,				//Condition LUT
	input							I_EnSrcA,			//Source-Input Enable
	input							I_EnSrcB,			//Source-Input Enable
	input	FTk_t		            I_OperandA,			//Source Operand
	input	FTk_t		            I_OperandB,			//Source Operand
	output	FTk_t		            O_Result,			//Result
	input	BTk_t					I_BTk,				//Backward Tokens
	output	BTk_t					O_BTkA,				//Backward Tokens
	output	BTk_t					O_BTkB				//Backward Tokens
);


	//// Logic Connect												////
	logic						Valid_AL;
	logic						Rls_AL;

	//	 Wait Unit
	BTk_t						BTk_A;
	BTk_t						BTk_B;
	BTk_t						BTk_C;

	FTk_t						W_FTk_A;
	FTk_t						W_FTk_B;

	FTk_t						SourceA_Add;
	FTk_t						SourceB_Add;
	FTk_t						SourceA_Logic;
	FTk_t						SourceB_Logic;

	BTk_t						BTk_Add;
	BTk_t						BTk_Logic;

	logic						SelLogic;

	logic						EnAdd;
	logic						EnLogic;

	FTk_t						ResultAdd;
	FTk_t						ResultLogic;
	FTk_t						Result;

	BTk_t						WBTk_Add;
	BTk_t						WBTk_Logic;

	FTk_t						FTk_AL;
	BTk_t						BTk_AL;

	FTk_t						SFTk_A;
	FTk_t						SFTk_B;


	//// Tokens														////
	//	 Valid Token
	assign Valid_AL			= ~( I_EnSrcA ^ W_FTk_A.v ) & ~( I_EnSrcB ^ W_FTk_B.v ) & I_Active;

	//	 Release Token
	assign Rls_AL			= ~( I_EnSrcA ^ W_FTk_A.r ) & ~( I_EnSrcB ^ W_FTk_B.r ) & I_Active;


	`ifdef EXTEND
	//// Skip Unit													////
	SyncUnit  SyncUnit_AL
	(
		.clock(				clock						),
		.reset(				reset						),
		.I_Active(			I_Active					),
		.I_EnSrcA(			I_EnSrcA					),
		.I_EnSrcB(			I_EnSrcB					),
		.I_FTk_A(			I_OperandA					),
		.I_FTk_B(			I_OperandB					),
		.O_BTk_A(			O_BTkA						),
		.O_BTk_B(			O_BTkB						),
		.O_FTk_A(			W_FTk_A						),
		.O_FTk_B(			W_FTk_B						),
		.I_BTk_A(			I_BTk						),
		.I_BTk_B(			I_BTk						),
		.O_SFTk_A(			SFTk_A						),
		.O_SFTk_B(			SFTk_B						),
		.O_Zero_A(										),
		.O_Zero_B(										),
		.O_Shared(										)
	);
	`else
	//// Input Pipeline Register									////
	assign W_BTkA.n			= BTk_A.n | Nack_A;
	assign W_BTkA.t			= BTk_A.t;
	assign W_BTkA.v			= BTk_A.v | W_FTk_A.r;
	assign W_BTkA.c			= BTk_A.c;

	assign W_BTkB.n			= BTk_B.n | Nack_B;
	assign W_BTkB.t			= BTk_B.t;
	assign W_BTkB.v			= BTk_B.v | W_FTk_B.r;
	assign W_BTkB.c			= BTk_B.c;

	DReg InReg0
	(
		.clock(				clock						),
		.reset(				reset						),
		.I_FTk(				I_FTk_A						),
		.O_BTk(				O_BTkA						),
		.I_We(				1'b1						),
		.O_FTk(				W_FTk_A						),
		.I_BTk(				WBk							)
	);

	DReg InReg1
	(
		.clock(				clock						),
		.reset(				reset						),
		.I_FTk(				I_FTk_B						),
		.O_BTk(				O_BTkB						),
		.I_We(				1'b1						),
		.O_FTk(				W_FTk_B						),
		.I_BTk(				WBk							)
	);

	assign SFTk_A			= '0;
	assign SFTk_B			= '0;


	//// Wait Unit													////
	WaitUnit WaitUnit_ML
	(
		.clock(				clock						),
		.reset(				reset						),
		.I_Active(			I_Active					),
		.I_En_A(			I_EnSrcA					),
		.I_En_B(			I_EnSrcB					),
		.I_Valid_A(			W_FTk_A.v					),
		.I_Valid_B(			W_FTk_B.v					),
		.I_Nack_A(			BTk_A.n						),
		.I_Nack_B(			BBk_B.n						),
		.I_Rls(				Result.r					),
		.O_Nack_A(			Nack_A						),
		.O_Nack_B(			Nack_B						)
	);
	`endif


	//// Execution Body												////
	assign SelLogic			= I_Opcode[6];

	//	 Source Distribution
	assign SourceA_Add		= ( SelLogic ) ? '0 :		W_FTk_A;
	assign SourceB_Add		= ( SelLogic ) ? '0 :		W_FTk_B;

	assign SourceA_Logic	= ( SelLogic ) ? W_FTk_A : '0;
	assign SourceB_Logic	= ( SelLogic ) ? W_FTk_B : '0;

	assign BTk_Add			= ( SelLogic ) ? '0 : 		BTk_C;
	assign BTk_Logic		= ( SelLogic ) ? BTk_C : 	'0;

	//	 Adder
	assign EnAdd			= Valid_AL & ~SelLogic;
	IntAddUnit IntAddUnit
	(
		.I_En(			EnAdd							),
		.I_Opcode(		I_Opcode						),
		.I_Cond(		I_Cond							),
		.I_OperandA(	SourceA_Add	    				),
		.I_OperandB(	SourceB_Add						),
		.O_Result(		ResultAdd						),
		.I_BTk(			BTk_Add			    			),
		.O_BTk(			WBTk_Add						)
	);

	//	 Logic
	assign EnLogic			= Valid_AL & SelLogic;
	LogicUnit LogicUnit
	(
		.I_En(			EnLogic							),
		.I_Opcode(		I_Opcode						),
		.I_Cond(		I_Cond							),
		.I_OperandA(	SourceA_Logic					),
		.I_OperandB(	SourceB_Logic					),
		.O_Result(		ResultLogic						),
		.I_BTk(			BTk_Logic						),
		.O_BTk(			WBTk_Logic						)
	);


	//// Output														////
	assign Result			= ( SelLogic ) ? ResultLogic : ResultAdd;


	DReg OutReg_AL
	(
		.clock(			clock							),
		.reset(			reset							),
		.I_FTk(			Result							),
		.I_BTk(			BTk_AL							),
		.I_We(			1'b1							),
		.O_FTk(			FTk_AL							),
		.O_BTk(			BTk_C							)
	);

	assign O_Result			= FTk_AL;
	assign BTk_AL			= I_BTk;

endmodule