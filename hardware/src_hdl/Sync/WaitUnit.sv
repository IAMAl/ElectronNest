///////////////////////////////////////////////////////////////////////////////////////////////////
//
//	ElectronNest
//	Copyright (C) 2024  Shigeyuki TAKANO
//
//  GNU AFFERO GENERAL PUBLIC LICENSE
//	version 3.0
//
//	Synchronization  Unit
//	Module Name:	WaitUnit
//	Function:
//					Synchronizing by Nack Token on Datapath
//
///////////////////////////////////////////////////////////////////////////////////////////////////

module WaitUnit
	import pkg_alu::*;
(
	input							clock,
	input							reset,
	input							I_Active,			//Flag:Active (Executable)
	input							I_En_A,				//Flag:Input-Source Enable
	input							I_En_B,				//Flag:Input-Source Enable
	input							I_Valid_A,			//Valid Token
	input							I_Valid_B,			//Valid Token
	input							I_Nack_A,			//Nack Token
	input							I_Nack_B,			//Nack Token
	input							I_Rls,  			//Release Token
	output	logic					O_Nack_A,			//Nack Token
	output	logic					O_Nack_B			//Nack Token
);

	fsm_wait					R_FSM_A;
	fsm_wait					R_FSM_B;

	logic						R_Ready_A;
	logic						R_Ready_B;

	logic						Valid_A;
	logic						Valid_B;
	logic						Valid;


	//// Valid Token												////
	//	 I_En:	Enable Source-Input
	//		0:	Disable Input
	//		1:	Enable Input
	assign Valid_A			= I_Active & ~( I_En_A ^ I_Valid_A );
	assign Valid_B			= I_Active & ~( I_En_B ^ I_Valid_B );
	assign Valid			= Valid_A & Valid_B;

	logic						R_Valid_A;
	logic						R_Valid_B;
	always_ff @( posedge clock ) begin: ff_valid_waitunit
		if ( reset ) begin
			R_Valid_A	<= 1'b0;
			R_Valid_B	<= 1'b0;
		end
		else begin
			R_Valid_A	<= Valid_A;
			R_Valid_B	<= Valid_B;
		end
	end

	logic						CancelWait;
	assign CancelWait		= ~R_Valid_A & ~R_Valid_B & ( R_FSM_A == wAIT_W ) & ( R_FSM_B == wAIT_W );


	//// Wait														////
	logic						Wait_A;
	logic						Wait_B;
	assign Wait_A			= ~Valid_B;
	assign Wait_B			= ~Valid_A;


	//// Nack at Nagetion of Source									////
	logic						Nack_ASAP_A;
	logic						Nack_ASAP_B;
	`ifdef EXTEND
	assign Nack_ASAP_A		= (( R_FSM_A == fILL_W ) | ~R_Ready_A ) & Valid_A & Wait_A;
	assign Nack_ASAP_B		= (( R_FSM_B == fILL_W ) | ~R_Ready_B ) & Valid_B & Wait_B;
	`else
	assign Nack_ASAP_A		= Valid_A & Wait_A;
	assign Nack_ASAP_B		= Valid_B & Wait_B;
	`endif


	//// Nack by Comming Nack Tokens								////
	logic						Nack;
	assign Nack				= I_Active & ( I_Nack_A | I_Nack_B );


	//// Wait State													////
	logic						is_Wait_A;
	logic						is_Wait_B;
	`ifdef EXTEND
	assign is_Wait_A		= ( R_FSM_A == wAIT_W ) & ( R_FSM_B != cHECK_W ) & ( R_FSM_B != wAIT_W ) & ~CancelWait;
	assign is_Wait_B		= ( R_FSM_B == wAIT_W ) & ( R_FSM_A != cHECK_W ) & ( R_FSM_A != wAIT_W )  & ~CancelWait;
	`else
	assign is_Wait_A		= ( R_FSM_A == wAIT_W ) & ~CancelWait;
	assign is_Wait_B		= ( R_FSM_B == wAIT_W ) & ~CancelWait;
	`endif

	logic						is_Check_A;
	logic						is_Check_B;
	`ifdef EXTEND
	assign is_Check_A		= ( R_FSM_A == cHECK_W ) & ( R_FSM_B == wAIT_W ) & Wait_A & Wait_B;
	assign is_Check_B		= ( R_FSM_B == cHECK_W ) & ( R_FSM_A == wAIT_W ) & Wait_B & Wait_A;
	`else
	assign is_Check_A		= 1'b0;
	assign is_Check_B		= 1'b0;
	`endif


	//// Output														////
	//	 Nack Token
	assign O_Nack_A			= Nack | Nack_ASAP_A | is_Wait_A | is_Check_A;
	assign O_Nack_B			= Nack | Nack_ASAP_B | is_Wait_B | is_Check_B;


	//// Synchronizing Control										////
	//		eMPTY_W
	//			Idling State, Data Word NOT Arrive
	//		fILL_W
	//			Must Capture Data Word
	//		wAIT_W
	//			Wait for Arriving Other Source
	//		cHECK_W
	//			Wait for Both Sources Arrive
	//
	//	cHECK State is necessary for Synchronizing
	//	If the state is not supported,
	//		one source should wait for other source,
	//		repeats the waiting for each other

	always_ff @( posedge clock ) begin: ff_fsm_waitunit_a
		if ( reset ) begin
			R_FSM_A	<= eMPTY_W;
		end
		else case ( R_FSM_A )
			eMPTY_W: begin
				if ( Valid_A & Valid_B ) begin
					R_FSM_A	<= fILL_W;
				end
				`ifdef EXTEND
				else if ( Valid_A & ~Valid_B ) begin
					R_FSM_A	<= wAIT_W;
				end
				`else
				else if ( Valid_A & ~Valid_B & ( R_FSM_B == eMPTY_W ) ) begin
					R_FSM_A	<= wAIT_W;
				end
				`endif
				else begin
					R_FSM_A <= eMPTY_W;
				end
			end
			fILL_W: begin
				if ( ~Valid_A & ~Valid_B ) begin
					R_FSM_A <= eMPTY_W;
				end
				else if ( Valid_A & Valid_B ) begin
					R_FSM_A <= fILL_W;
				end
				else if ( Valid_A & ~Valid_B ) begin
					if ( R_FSM_B == cHECK_W  ) begin
						R_FSM_A	<= cHECK_W;
					end
					else begin
						R_FSM_A	<= wAIT_W;
					end
				end
				else if ( ~Valid_A & Valid_B ) begin
					R_FSM_A	<= eMPTY_W;
				end
				else begin
					R_FSM_A	<= R_FSM_A;
				end
			end
			wAIT_W: begin
				if ( Valid_B ) begin
					R_FSM_A	<= cHECK_W;
				end
				else if ( CancelWait ) begin
					R_FSM_A	<= cHECK_W;
				end
				else begin
					R_FSM_A	<= wAIT_W;
				end
			end
			cHECK_W: begin
				if ( ~Valid_A & ~Valid_B ) begin
					R_FSM_A	<= eMPTY_W;
				end
				else if ( Valid_A & Valid_B ) begin
					R_FSM_A	<= fILL_W;
				end
				else if ( Valid_A & ~Valid_B ) begin
					if ( R_FSM_B == fILL_W  ) begin
						R_FSM_A	<= cHECK_W;
					end
					else begin
						R_FSM_A	<= wAIT_W;
					end
				end
				else if ( ~Valid_A & Valid_B ) begin
					R_FSM_A	<= wAIT_W;
				end
				else begin
					R_FSM_A	<= R_FSM_A;
				end
			end
		endcase
	end


	always_ff @( posedge clock ) begin: ff_fsm_waitunit_b
		if ( reset ) begin
			R_FSM_B	<= eMPTY_W;
		end
		else case ( R_FSM_B )
			eMPTY_W: begin
				if ( Valid_B & Valid_A ) begin
					R_FSM_B	<= fILL_W;
				end
				`ifdef EXTEND
				else if ( Valid_B & ~Valid_A ) begin
					R_FSM_B	<= wAIT_W;
				end
				`else
				else if ( Valid_B & ~Valid_A & ( R_FSM_A == eMPTY_W ) ) begin
					R_FSM_B	<= wAIT_W;
				end
				`endif
				else begin
					R_FSM_B <= eMPTY_W;
				end
			end
			fILL_W: begin
				if ( ~Valid_B & ~Valid_A ) begin
					R_FSM_B <= eMPTY_W;
				end
				else if ( Valid_B & Valid_A ) begin
					R_FSM_B <= fILL_W;
				end
				else if ( Valid_B & ~Valid_A ) begin
					if ( R_FSM_A == cHECK_W  ) begin
						R_FSM_B	<= cHECK_W;
					end
					else begin
						R_FSM_B	<= wAIT_W;
					end
				end
				else if ( ~Valid_B & Valid_A ) begin
					R_FSM_B	<= eMPTY_W;
				end
				else begin
					R_FSM_B	<= R_FSM_B;
				end
			end
			wAIT_W: begin
				if ( Valid_A ) begin
					R_FSM_B	<= cHECK_W;
				end
				else if ( CancelWait ) begin
					R_FSM_B	<= cHECK_W;
				end
				else begin
					R_FSM_B	<= wAIT_W;
				end
			end
			cHECK_W: begin
				if ( ~Valid_B & ~Valid_A ) begin
					R_FSM_B	<= eMPTY_W;
				end
				else if ( Valid_B & Valid_A ) begin
					R_FSM_B	<= fILL_W;
				end
				else if ( Valid_B & ~Valid_A ) begin
					if ( R_FSM_A == fILL_W  ) begin
						R_FSM_B	<= cHECK_W;
					end
					else begin
						R_FSM_B	<= wAIT_W;
					end
				end
				else if ( ~Valid_B & Valid_A ) begin
					R_FSM_B	<= wAIT_W;
				end
				else begin
					R_FSM_B	<= R_FSM_B;
				end
			end
		endcase
	end


	//// Ready Flag													////
	//	 First Data Word Arrive-Detection
	always_ff @( posedge clock ) begin: ff_ready_a_waitunit
		if ( reset ) begin
			R_Ready_A	<= 1'b0;
		end
		else if ( I_Rls ) begin
			R_Ready_A	<= 1'b0;

		end
		else if ( Valid_A ) begin
			R_Ready_A	<= 1'b1;
		end
		else begin
			R_Ready_A	<= R_Ready_A;
		end
	end

	always_ff @( posedge clock ) begin: ff_ready_b_waitunit
		if ( reset ) begin
			R_Ready_B	<= 1'b0;
		end
		else if ( I_Rls ) begin
			R_Ready_B	<= 1'b0;

		end
		else if ( Valid_B ) begin
			R_Ready_B	<= 1'b1;
		end
		else begin
			R_Ready_B	<= R_Ready_B;
		end
	end

endmodule
