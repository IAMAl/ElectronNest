///////////////////////////////////////////////////////////////////////////////////////////////////
//
//	ElectronNest
//	Copyright (C) 2024  Shigeyuki TAKANO
//
//  GNU AFFERO GENERAL PUBLIC LICENSE
//	version 3.0
//
//	Interface Logic Circuit
//	Module Name:	IFLogic
//	Function:
//					Interfacing between ComputeTile and Ritch IF Logic
//					Connecting betwen Element and Buffer or Element and Ext Memory IF
//
///////////////////////////////////////////////////////////////////////////////////////////////////

module IFLogic
	import pkg_en::*;
	import pkg_bram_if::*;
(
	input							clock,
	input							reset,
	input							I_Ld,				//Flag: Loading
	input							I_St,				//Flag: Storing
	input	FTk_t					I_FTk_IF,			//From ERAM
	output	BTk_t					O_BTk_IF,			//To ERAM
	output	FTk_t		            O_FTk_IF,			//To ERAM
	input	BTk_t					I_BTk_IF,			//From ERAM
	output	FTk_t					O_FTk,				//To Compute Tile
	input	BTk_t					I_BTk,				//From Compute Tile
	input	FTk_t					I_FTk,				//From Compute Tile
	output	BTk_t					O_BTk,				//to Compute Tile
	output	FTk_t					O_Req_FTk,			//Request to Front-End
	output	logic					O_Header			//Flag: Send Request
);


	logic							acq_message;
	logic							acq_flagmsg;
	logic							rls_message;
	logic							rls_flagmsg;
	logic							is_Acq;
	logic							is_Rls;

	logic							R_Ld;
	logic							R_St;

	logic							Header;
	logic							Nack_to_CU;

	FTk_t							W_I_FTk_IF;
	BTk_t							W_I_BTk_IF;
	FTk_t							W_O_FTk_IF;
	BTk_t							W_O_BTk_IF;

	FTk_t							W_I_FTk;
	BTk_t							W_I_BTk;
	FTk_t							W_O_FTk;
	BTk_t							W_O_BTk;

	FTk_t							B_L_FTk_IF;
	BTk_t							B_L_BTk_IF;
	FTk_t							B_S_FTk_IF;
	BTk_t							B_S_BTk_IF;

	logic							Ld_Buff_Empty;
	logic							Ld_Buff_Full;

	logic							Charge_St;
	logic							Release_St;
	logic							St_Buff_Empty;
	logic							St_Buff_Full;

	logic							St_Valid;
	logic							is_RConfigData;
	logic							is_RoutingData;

	logic							Ld_Term;
	logic							St_Term;

	FTk_t							H_I_FTk;
	BTk_t							W_H_BTk;
	FTk_t							B_FTk;

	logic	[WIDTH_LENGTH-1:0]		Length;

	logic							Set_RConfig;

	logic							Start_BackEnd;

	fsm_iflogic_st_frontend			FSM_FrontEnd;
	fsm_iflogic_st_backend			FSM_BackEnd;

	logic							R_is_Acq;

	logic							R_Ld_Acvtive;

	logic							St_Nack_BackEnd;

	logic	[3:0]					CNT;
	always_ff @( posedge clock ) begin
		if ( reset ) begin
			CNT	<= 0;
		end
		else if ( I_FTk_IF.v & I_FTk_IF.a & ~I_FTk_IF.r & ~( O_FTk.v & O_FTk.a & O_FTk.r ) ) begin
			CNT	<= CNT + 1;
		end
		else if ( ~( I_FTk_IF.v & I_FTk_IF.a & ~I_FTk_IF.r ) & O_FTk.v & O_FTk.a & O_FTk.r ) begin
			CNT	<= CNT - 1;
		end
	end


	//// Load-Path ( Extern -> CU)									////
	//	 Send to CU
	assign O_FTk				= ( R_Ld_Acvtive ) ?	W_O_FTk_IF	: '0;

	assign W_I_BTk_IF.n			= ( R_Ld_Acvtive ) ?  	I_BTk.n	: '0;
	assign W_I_BTk_IF.t			= ( R_Ld_Acvtive ) ?  	I_BTk.t : '0;
	assign W_I_BTk_IF.v			= ( R_Ld_Acvtive ) ?  	I_BTk.v	: '0;
	assign W_I_BTk_IF.c			= ( R_Ld_Acvtive ) ?  	I_BTk.c	: '0;

	//Ld-DReg <-> Ld-Buff
	assign W_I_FTk_IF			= B_L_FTk_IF;
	assign B_L_BTk_IF			= W_O_BTk_IF;

	assign Ld_Term				= ( W_O_FTk_IF.v & W_O_FTk_IF.a & W_O_FTk_IF.r ) | O_BTk.t;


	//// Store-Path (CU -> Extern)									////
	assign St_Valid				= H_O_FTk.v;

	//Reduest to FrontEnd (IFUnit)
	assign Header				= ( ( FSM_FrontEnd < IFLOGIC_ST_FRONT_RUN ) & ( FSM_FrontEnd > IFLOGIC_ST_INIT ) ) | ( ( FSM_FrontEnd == IFLOGIC_ST_INIT ) & is_Acq );
	assign O_Header				= Header;
	assign O_Req_FTk			= ( Header ) ? 				W_O_FTk :
															'0;

	//St-Buff <-> St-DReg;
	assign H_I_FTk.v			= H_O_FTk.v;
	assign H_I_FTk.a			= H_O_FTk.a | ( ( FSM_FrontEnd == IFLOGIC_ST_CHECK_ATTRIB ) & St_Valid & ( ~is_RConfigData | ( is_RConfigData & ( I_FTk.d[15:8] == 8'h3 ) ) ) ) | ( ( FSM_FrontEnd == IFLOGIC_ST_SEND_RCFG ) & St_Valid );
	assign H_I_FTk.c			= H_O_FTk.c;
	assign H_I_FTk.r			= H_O_FTk.r | ( ( FSM_FrontEnd == IFLOGIC_ST_CHECK_ATTRIB ) & St_Valid & ( ~is_RConfigData | ( is_RConfigData & ( I_FTk.d[15:8] == 8'h3 ) ) ) ) | ( ( FSM_FrontEnd == IFLOGIC_ST_SEND_RCFG ) & St_Valid );
	assign H_I_FTk.d			= H_O_FTk.d;
	assign H_I_FTk.i			= H_O_FTk.i;

	logic						Send_FTk_Req;
	assign Send_FTk_Req			= ( ( FSM_FrontEnd < IFLOGIC_ST_SEND_ROUTE ) & ( FSM_FrontEnd > IFLOGIC_ST_INIT ) ) | ( ( FSM_FrontEnd == IFLOGIC_ST_INIT ) & is_Acq );
	assign W_I_FTk				= ( FSM_BackEnd == IFLOGIC_ST_BACK_RUN ) ?	B_S_FTk_IF :
									( Send_FTk_Req ) ? 						H_I_FTk :
																			'0;

	assign W_H_BTk.n			= Nack_to_CU;
	assign W_H_BTk.t			= 1'b0;
	assign W_H_BTk.v			= 1'b0;
	assign W_H_BTk.c			= 1'b0;
	assign B_S_BTk_IF			= ( FSM_BackEnd == IFLOGIC_ST_BACK_RUN ) ?	W_O_BTk :
																			W_H_BTk;

	//St-Dreg <-> Extern
	assign O_FTk_IF				= ( R_St ) ?	W_O_FTk :	'0;
	assign W_I_BTk				= ( R_St ) ?	I_BTk_IF :	'0;

	//Tokens
	assign is_Acq				= acq_message | acq_flagmsg;
	assign is_Rls				= rls_message | rls_flagmsg;

	//Nack to CU
	assign Nack_to_CU			= ( Set_RConfig | ( FSM_BackEnd == IFLOGIC_ST_BACK_RUN ) ) & ~R_St;

	//Store-Buffer

	assign Release_St			= ( FSM_BackEnd == IFLOGIC_ST_BACK_RUN );

	assign St_Term				= ( W_O_FTk.v & W_O_FTk.a & W_O_FTk.r );

	//Capture Load and Store Path Established
	always_ff @( posedge clock ) begin
		if ( reset ) begin
			R_Ld				<= 1'b0;
			R_St				<= 1'b0;
		end
		else begin
			R_Ld				<= I_Ld;
			R_St				<= I_St;
		end
	end


	//// Load Path Controller										////
	always_ff @( posedge clock ) begin
		if ( reset ) begin
			R_is_Acq		<= 1'b0;
		end
		else if ( is_Rls ) begin
			R_is_Acq		<= 1'b0;
		end
		else if ( is_Acq ) begin
			R_is_Acq		<= 1'b1;
		end
	end

	always_ff @( posedge clock ) begin
		if ( reset ) begin
			R_Ld_Acvtive	<= 1'b0;
		end
		else if ( ( Ld_Term & ~R_is_Acq ) | ~R_Ld & O_FTk.r ) begin
			R_Ld_Acvtive	<= 1'b0;
		end
		else if ( I_FTk_IF.v & I_FTk_IF.a & ~I_FTk_IF.r ) begin
			R_Ld_Acvtive	<= 1'b1;
		end
	end

	always_ff @( posedge clock ) begin
		if ( reset ) begin
			Set_RConfig		<= 1'b0;
		end
		else if ( FSM_FrontEnd == 0 ) begin
			Set_RConfig		<= 1'b0;
		end
		else if ( ( FSM_FrontEnd == 5 ) & St_Valid & is_RConfigData & ( Length == 0 ) ) begin
			Set_RConfig		<= 1'b1;
		end
	end


	//// Store Path Controller										////
	//	 Store Front-End
	always_ff @( posedge clock ) begin
		if ( reset ) begin
			FSM_FrontEnd		<= IFLOGIC_ST_INIT;
		end
		else case ( FSM_FrontEnd )
		IFLOGIC_ST_INIT: begin
			if ( is_Acq ) begin
				//Capture My-ID
				FSM_FrontEnd	<= IFLOGIC_ST_ID_T;
			end
			else begin
				FSM_FrontEnd	<= IFLOGIC_ST_INIT;
			end
		end
		IFLOGIC_ST_ID_T: begin
			if ( St_Valid ) begin
				//Capture T-ID, Send My-ID
				FSM_FrontEnd	<= IFLOGIC_ST_ID_F;
			end
		end
		IFLOGIC_ST_ID_F: begin
			if ( St_Valid ) begin
				//Capture F-ID, Send T-ID
				FSM_FrontEnd	<= IFLOGIC_ST_ATTRIB;
			end
		end
		IFLOGIC_ST_ATTRIB: begin
			if ( St_Valid ) begin
				//Capture Attrib Word, Send F-ID
				FSM_FrontEnd	<= IFLOGIC_ST_ROUTE;
			end
		end
		IFLOGIC_ST_ROUTE: begin
			if ( St_Valid ) begin
				//Capture Route Data, Send Attrib Word
				FSM_FrontEnd	<= IFLOGIC_ST_CHECK_ATTRIB;
			end
		end
		IFLOGIC_ST_CHECK_ATTRIB: begin
			if ( St_Valid & is_RConfigData & ( Length == 0 ) ) begin
				//Capture Attrib Word, Send Route Data
				FSM_FrontEnd	<= IFLOGIC_ST_CHECK_RCFG;
			end
			else if ( St_Valid ) begin
				//Capture Attrib Word, Send Route Data
				FSM_FrontEnd	<= IFLOGIC_ST_SEND_ROUTE;
			end
		end
		IFLOGIC_ST_CHECK_RCFG: begin
			if ( St_Valid ) begin
				//Capture R-COnfig Data for BRAM, Send Attrib Word
				FSM_FrontEnd	<= IFLOGIC_ST_SEND_RCFG;
			end
		end
		IFLOGIC_ST_SEND_RCFG: begin
			if ( St_Valid ) begin
				//Send R-Config Data
				FSM_FrontEnd	<= IFLOGIC_ST_RETIME_RCFG;
			end
		end
		IFLOGIC_ST_RETIME_RCFG: begin
				//Send R-Config Data
				FSM_FrontEnd	<= IFLOGIC_ST_FRONT_RUN;
		end
		IFLOGIC_ST_SEND_ROUTE: begin
				FSM_FrontEnd	<= IFLOGIC_ST_FRONT_RUN;
		end
		IFLOGIC_ST_FRONT_RUN: begin
			if ( R_St ) begin
				//Start BackEnd
				FSM_FrontEnd	<= IFLOGIC_ST_INIT;
			end
		end
		default: begin
			FSM_FrontEnd		<= IFLOGIC_ST_INIT;
		end
		endcase
	end

	//	 Store BackEnd
	assign Start_BackEnd	= ( FSM_FrontEnd == IFLOGIC_ST_FRONT_RUN ) & I_St;
	always_ff @( posedge clock ) begin
		if ( reset ) begin
			FSM_BackEnd			<= IFLOGIC_ST_BACK_INIT;
		end
		else case ( FSM_BackEnd )
		IFLOGIC_ST_BACK_INIT: begin
			if ( Start_BackEnd ) begin
				FSM_BackEnd		<= IFLOGIC_ST_BACK_RUN;
			end
			else begin
				FSM_BackEnd		<=IFLOGIC_ST_BACK_INIT;
			end
		end
		IFLOGIC_ST_BACK_RUN: begin
			if ( St_Term ) begin
				FSM_BackEnd		<=IFLOGIC_ST_BACK_INIT;
			end
		end
		default: begin
			FSM_BackEnd			<= IFLOGIC_ST_BACK_INIT;
		end
		endcase
	end


	//// Tokken Detection											////
	TokenDec TokenDec (
		.I_FTk(				I_FTk						),
		.O_acq_message(		acq_message					),
		.O_rls_message(		rls_message					),
		.O_acq_flagmsg(		acq_flagmsg					),
		.O_rls_flagmsg(		rls_flagmsg					)
	);


	//// I/O Register												////
	//	Input from Extern
	DReg Ld_DReg_IF (
		.clock(				clock						),
		.reset(				reset						),
		.I_We(				1'b1						),
		.I_FTk(				W_I_FTk_IF					),
		.O_BTk(				W_O_BTk_IF					),
		.O_FTk(				W_O_FTk_IF					),
		.I_BTk(				W_I_BTk_IF					)
	);

	//	 Input from Compute Tile
	FTk_t					H_O_FTk;
	BTk_t					H_O_BTk;
	DReg St_DReg_Retime_IF (
		.clock(				clock						),
		.reset(				reset						),
		.I_We(				1'b1						),
		.I_FTk(				I_FTk						),
		.O_BTk(				H_O_BTk						),
		.O_FTk(				H_O_FTk						),
		.I_BTk(				I_BTk						)
	);

	DReg St_DReg_IF (
		.clock(				clock						),
		.reset(				reset						),
		.I_We(				1'b1						),
		.I_FTk(				W_I_FTk						),
		.O_BTk(				W_O_BTk						),
		.O_FTk(				W_O_FTk						),
		.I_BTk(				W_I_BTk						)
	);


	//// Buffer														////
	//	 Buffer for Data from External World
	Buff_IF #(
		.DEPTH_FIFO(		DEPTH_FIFO					),
		.THRESHOLD(			6							),
		.TYPE_FWRD(			FTk_t						)
	) LdBuff
	(
		.clock(				clock						),
		.reset(				reset						),
		.I_FTk(				I_FTk_IF					),
		.O_BTk(				O_BTk_IF					),
		.O_FTk(				B_L_FTk_IF					),
		.I_BTk(				B_L_BTk_IF					),
		.I_We(				1'b1						),
		.I_Re(				1'b1						),
		.O_Empty(			Ld_Buff_Empty				),
		.O_Full(			Ld_Buff_Full				)
	);

	//	 Buffer for Data from Compute Tile
	FTk_t					W_FTk_Req;
	assign W_FTk_Req.v		= I_FTk.v;
	assign W_FTk_Req.a		= 1'b1;
	assign W_FTk_Req.c		= I_FTk.c;
	assign W_FTk_Req.r		= 1'b1;
	assign W_FTk_Req.i		= I_FTk.i;
	assign W_FTk_Req.d		= I_FTk.d;
	assign B_FTk		= //( ( FSM_FrontEnd > IFLOGIC_ST_ID_F ) & ( FSM_FrontEnd < IFLOGIC_ST_CHECK_ATTRIB ) ) ?				'0 :
							( ( FSM_FrontEnd == IFLOGIC_ST_CHECK_ATTRIB ) & St_Valid & is_RConfigData & ( Length == 0 )) ?	'0 :
							//( FSM_FrontEnd == IFLOGIC_ST_SEND_RCFG ) ?														'0 :
							( FSM_FrontEnd == IFLOGIC_ST_RETIME_RCFG ) ?													'0 :
							//( FSM_FrontEnd == IFLOGIC_ST_SEND_ROUTE ) ?														W_FTk_Req :
																															I_FTk;
	logic					We_BuffSt;
	logic					Re_BuffSt;
	assign Charge_St	= ( ( FSM_FrontEnd == IFLOGIC_ST_INIT ) & is_Acq ) | (( FSM_FrontEnd > IFLOGIC_ST_INIT ) & ( FSM_FrontEnd < IFLOGIC_ST_ATTRIB ));
	assign We_BuffSt	= Charge_St |
							( is_RConfigData & ( I_FTk.d[15:8] == 8'h3 ) ) |
							( FSM_FrontEnd == IFLOGIC_ST_SEND_ROUTE ) |
							( FSM_FrontEnd == IFLOGIC_ST_FRONT_RUN ) |
							( FSM_BackEnd == IFLOGIC_ST_BACK_RUN );
	assign Re_BuffSt	= ~( ( FSM_FrontEnd == IFLOGIC_ST_FRONT_RUN ) & ~R_St ) & ( FSM_FrontEnd != IFLOGIC_ST_SEND_ROUTE ) & ~( ( FSM_FrontEnd > IFLOGIC_ST_ID_F ) & ( FSM_FrontEnd < IFLOGIC_ST_SEND_ROUTE ) ) & ~Charge_St;
	Buff_IF #(
		.DEPTH_FIFO(		DEPTH_FIFO					),
		.THRESHOLD(			6							),
		.TYPE_FWRD(			FTk_t						)
	) StBuff
	(
		.clock(				clock						),
		.reset(				reset						),
		.I_FTk(				B_FTk						),
		.O_BTk(				O_BTk						),
		.O_FTk(				B_S_FTk_IF					),
		.I_BTk(				B_S_BTk_IF					),
		.I_We(				We_BuffSt					),
		.I_Re(				Re_BuffSt					),
		.O_Empty(			St_Buff_Empty				),
		.O_Full(			St_Buff_Full				)
	);

	AttributeDec IfLogic_AttributeDec_Retime
	(
		.I_Data(			H_O_FTk.d					),
		.is_Pull(										),
		.is_DataWord(									),
		.is_RConfigData(								),
		.is_PConfigData(								),
		.is_RoutingData(	is_RoutingData				),
		.is_Shared(										),
		.is_NonZero(									),
		.is_Dense(										),
		.is_MyAttribute(								),
		.is_Term_Block(									),
		.is_In_Cond(									),
		.O_Length(										)
	);

	AttributeDec IfLogic_AttributeDec
	(
		.I_Data(			I_FTk.d						),
		.is_Pull(										),
		.is_DataWord(									),
		.is_RConfigData(	is_RConfigData				),
		.is_PConfigData(								),
		.is_RoutingData(								),
		.is_Shared(										),
		.is_NonZero(									),
		.is_Dense(										),
		.is_MyAttribute(								),
		.is_Term_Block(									),
		.is_In_Cond(									),
		.O_Length(			Length						)
	);

endmodule