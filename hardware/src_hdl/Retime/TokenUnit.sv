///////////////////////////////////////////////////////////////////////////////////////////////////
//
//	ElectronNest
//	Copyright (C) 2024  Shigeyuki TAKANO
//
//  GNU AFFERO GENERAL PUBLIC LICENSE
//	version 3.0
//
//	Token Control Unit
//	Module Name:	TokenUnit
//	Function:
//					Retiming Control by Tokens of
//					Valid Token
//					Acq Token
//					Release Token
//					Nack Token
//
///////////////////////////////////////////////////////////////////////////////////////////////////

module TokenUnit
	import  pkg_en::*;
(
	input							clock,
	input							reset,
	input	logic					I_Valid,			//Valid Token
	input	logic					I_Nack,				//Nack Token
	input	logic					I_We,				//Write-Enable
	output	logic					O_Valid,			//Valid Token
	output	logic					O_Nack,				//Nack Token
	output	logic					O_We,				//Write-Enable
	output	logic					O_WNo,				//Write Context No.
	output	logic					O_RNo				//Read Context No.
);

	logic						Valid_Run;
	logic						Stall;

	logic						is_eMPTY;
	logic						is_fILL;
	logic						is_wAIT;
	logic						is_rEVERT;

	logic						We;
	logic						Re;
	logic						FlipW;
	logic						FlipR;

	logic						Empty;
	logic						NoNack;
	logic						En_Vld;

	logic						Recover;

	logic	[1:0]				Num;

	fsm_token					R_FSM;
	logic						R_Valid [1:0];
	logic						R_Nack;
	logic						R_NackD1;
	logic						R_NackD2;
	logic						R_Revert;
	logic						R_WNo;
	logic						R_RNo;


	logic						Pulse_Nack_Bar;
	logic						Pulse_Nack;

	//// Control Status												////
	assign is_eMPTY			= ( R_FSM == eMPTY );
	assign is_fILL			= ( R_FSM == fILL );
	assign is_wAIT			= ( R_FSM == wAIT );
	assign is_rEVERT		= ( R_FSM == rEVERT );

	//	 Pulse of nack detection
	//	 used for corner case
	assign Pulse_Nack		= ~I_Nack & & R_NackD1 & ~R_NackD2;

	//// Combination Table
	//	I_Valid	I_Nack NoNack	Valid_Run	Stall
	//	      0      0      0           0       0
	//        0      1      1           0       0
	//        1      0      0           1       0
	//        1      1      0           0       1


	//// Enable to Sending											////
	assign Valid_Run		= I_Valid & ~I_Nack;


	//// Stall Signal												////
	//	 Forcing Back-up the Input
	assign Stall			= I_Valid & I_Nack;


	//// Not Captured												////
	assign Empty			= ~I_Valid & ~I_Nack;


	//// Not Send Nack Token										////
	//	 also used for write-enable when nack arrives at fILL state
	assign NoNack			= ~I_Valid & I_Nack;


	//// Enable to output Valid	Token								////
	assign En_Vld			= ( is_eMPTY & R_Revert )		|
								is_fILL						|
								( is_wAIT & Pulse_Nack )	|
								( is_rEVERT & ~R_Nack );


	//// Enable to Capture											////
	//	 Write-enable
	assign We				= I_Valid | ( is_fILL & NoNack );

	//	 Read-enable
	assign Re				= is_fILL | is_rEVERT;


	//// Flip Capturing Register No.								////
	assign FlipW			= ( is_eMPTY & Stall ) | ( is_fILL & I_Nack );

	assign FlipR			= ( is_rEVERT ) |
								( is_wAIT & Pulse_Nack ) |
								( is_eMPTY & R_Revert & R_Nack & ~I_Nack );


	//// Outputs													////
	assign O_Valid			= ( R_Valid[ R_RNo ] & En_Vld );
	assign O_We				= We & ( is_wAIT | I_We );
	assign O_Nack			= R_Nack;
	assign O_WNo			= R_WNo;
	assign O_RNo			= R_RNo;


	//// Pulse Nack-Bar Detection
	//	 used for coner case
	always_ff @( posedge clock ) begin
		if ( reset ) begin
			Pulse_Nack_Bar	<= 1'b0;
		end
		else if ( ~I_Nack ) begin
			Pulse_Nack_Bar	<= 1'b0;
		end
		else if ( Stall & ~R_NackD1 & R_NackD2 ) begin
			Pulse_Nack_Bar	<= 1'b1;
		end
	end

	//// Store Valid Token											////
	//	 Capture dropped Valid Token
	assign Recover			= Num[1] | ( ( Num == 2'h0 ) & R_NackD2 & ~I_Valid & R_Valid[ R_WNo ] );
	always_ff @( posedge clock ) begin: ff_valid_tokenunit
		if ( reset ) begin
			R_Valid[0]			<= 1'b0;
			R_Valid[1]			<= 1'b0;
		end
		else if ( We | is_rEVERT | Pulse_Nack | FlipR ) begin
			if ( We | is_rEVERT | Pulse_Nack ) begin
				R_Valid[ R_WNo ]	<= I_Valid | Recover;
			end
			else if ( FlipR ) begin
				R_Valid[ R_RNo ]	<= Pulse_Nack_Bar;
			end
		end
	end


	//// Capture NOT when Empty										////
	always_ff @( posedge clock ) begin: ff_nack_tokenunit
		if ( reset ) begin
			R_Nack			<= 1'b0;
			R_NackD1		<= 1'b0;
			R_NackD2		<= 1'b0;
		end
		else begin
			R_Nack  		<= I_Nack & (
								( ~is_eMPTY	& (  I_We | ~I_Valid ))	|
								(  is_eMPTY & ( ~I_We |  I_Valid ))
							);
			R_NackD1		<= I_Nack;
			R_NackD2		<= R_NackD1;
		end
	end


	//// Flip Select No.											////
	//	 Flip Write No
	always_ff @( posedge clock ) begin: ff_wno_tokenunit
		if ( reset ) begin
			R_WNo			<= 1'b0;
		end
		else if ( FlipW ) begin
			R_WNo			<= ~R_WNo;
		end
	end

	//	 Flip Read No
	always_ff @( posedge clock ) begin: ff_rno_tokenunit
		if ( reset ) begin
			R_RNo			<= 1'b0;
		end
		else if ( FlipR | is_rEVERT ) begin
			R_RNo			<= R_WNo;
		end
	end


	//// Control FSM												////
	always_ff @( posedge clock ) begin: ff_fsm_tokenunit
		if ( reset ) begin
			R_FSM			<= eMPTY;
			R_Revert		<= 1'b0;
		end
		else case ( R_FSM )
			eMPTY: begin
				if ( Stall ) begin
					R_FSM		<= wAIT;
					R_Revert	<= 1'b1;
				end
				else if ( I_Valid ) begin
					R_FSM		<= fILL;
					R_Revert	<= 1'b0;
				end
				else begin
					R_FSM		<= eMPTY;
					R_Revert	<= 1'b0;
				end
			end
			fILL: begin
				if ( Valid_Run ) begin
					R_FSM		<= fILL;
					R_Revert	<= 1'b0;
				end
				else if ( Stall ) begin
					R_FSM		<= wAIT;
					R_Revert	<= 1'b1;
				end
				else if ( NoNack ) begin
					R_FSM		<= wAIT;
					R_Revert	<= 1'b1;
				end
				else if ( Empty ) begin
					R_FSM		<= eMPTY;
					R_Revert	<= 1'b0;
				end
			end
			wAIT: begin
				if ( Valid_Run ) begin
					R_FSM		<= rEVERT;
					R_Revert	<= 1'b0;
				end
				else if ( NoNack ) begin
					R_FSM		<= wAIT;
					R_Revert	<= 1'b0;
				end
				else if ( Empty ) begin
					R_FSM		<= rEVERT;
					R_Revert	<= 1'b0;
				end
			end
			rEVERT: begin
				if ( I_Valid & ~R_Revert ) begin
					R_FSM		<= fILL;
					R_Revert	<= 1'b0;
				end
				else if ( I_Valid & R_Revert ) begin
					R_FSM		<= wAIT;
					R_Revert	<= 1'b1;
				end
				else if ( NoNack ) begin
					R_FSM		<= rEVERT;
					R_Revert	<= 1'b0;
				end
				else begin
					R_FSM		<= eMPTY;
					R_Revert	<= 1'b1;
				end
			end
			default: begin
					R_FSM		<= eMPTY;
					R_Revert	<= 1'b0;
			end
		endcase
	end

	always_ff @( posedge clock ) begin: ff_count_tokenunit
		if ( reset ) begin
			Num		<= 2'h0;
		end
		else if ( We & ~Re ) begin
			Num		<= Num + 1'b1;
		end
		else if ( ~We & Re ) begin
			Num		<= Num - 1'b1;
		end
	end

endmodule
