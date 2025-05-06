///////////////////////////////////////////////////////////////////////////////////////////////////
//
//	ElectronNest
//	Copyright (C) 2024  Shigeyuki TAKANO
//
//  GNU AFFERO GENERAL PUBLIC LICENSE
//	version 3.0
//
//	Port on ALU
//	Module Name:	PortSync
//	Function:
//					Synchronization on Port
//					Wait for Ariiving Other Operand by Issueing Nack Token.
//
///////////////////////////////////////////////////////////////////////////////////////////////////

module PortSync
	import	pkg_alu::*;
#(
	parameter int WIDTH_LENGTH		= 10
)(
	input							clock,
	input							reset,
	input	[WIDTH_LENGTH-1:0]		I_Length,			//Attribute Block Length
	input							I_Valid,			//Valid Token
	input							I_Valid_Other1,		//Valid Token
	input							I_Valid_Other2,		//Valid Token
	input							I_Acq,				//Acq Token
	input							I_Rls,				//Release Token
	input							I_Nack,				//Nack Token
	input							I_Done,				//Flag: Exec is Done
	input							is_Configured,		//Flag: Already configured by other
	input							is_Ready1,			//Flag: Ready to Data-Feed
	input							is_Ready2,			//Flag: Ready to Data-Feed
	input							is_PConfigData,		//Flag: Attirb Block is P-Config
	input							is_RConfigData,		//Flag: Attrib Block is R-Config
	input							is_RouteData,		//Flag: Attrib Block is Routing Data
	input							is_AuxData,			//Flag: Attrib Block is Data
	output	logic					O_Ready,			//Flag: Ready to Data-Feed
	output	logic					O_Bypass,			//Flag: Bypassing
	output	logic					O_SendHead,			//Flag: Send Header (IDs)
	output	logic					O_InputData,		//Feed Data
	output	logic					O_SetDAttrib,		//Set Attribute Word
	output	logic					O_Nack_My,			//Nack Token
	output	[WIDTH_LENGTH-1:0]		O_Length,			//Attribute Block Length
	output	logic					O_Configure,		//Do Configuration
	output	logic					O_SelPort,			//Select Port
	output	logic					O_Backup_Attrib,	//Backup Attribute Word
	output	logic					O_Set_Backup,		//Use of Backed up Attribute Word
	output	logic					O_is_Valid			//Flag: Valid Token
);

	fsm_port					R_FSM;					//FSM for Sync of Operands


	//// Logic Connect												////
	logic						ZeroLength;				//Flag: Actual Length is "One"
	logic						AttribData;				//Flag: Attrib for First Arrived Data Block
	logic						Wait;					//Flag:	Waiting for Source Operand
	logic						Config_Other;			//Flag: Configured by Other
	logic						Backup_Attrib;			//Backup: Instruction

	logic						is_FSM_Check_Attirb;

	logic						is_no_ready;
	logic						is_ready;


	//// Capture Signal												////
	logic						R_Valid;
	logic						R_Valid_Other1;
	logic						R_Valid_Other2;
	logic						R_Nack;
	logic						R_Rls;

	logic						R_Config_by_Port;
	logic						R_Ready;
	logic						R_Bypass;

	logic	[WIDTH_LENGTH-1:0]	R_Length;				//Attribution Block Length
	logic						R_InputData;			//Flag:	Feed Operand
	logic						R_SetPConfig;			//Flag:	Set PE-Config
	logic						R_SetDAttrib;			//Flag:	Set Data Attrib Block
	logic						R_Configure;			//Flag: Do Configuration
	logic						R_DataWord;				//Flag:	Attrib Block is Data Bloxk
	logic						R_ConfigDone;			//Flag:	Configuration is done by this operand
	logic						R_Set_Backup;			//Flag: Backing Up Attribute Word


	//	 Capture Input
	always_ff @( posedge clock ) begin: ff_token_aluport
		if ( reset ) begin
			R_Valid			<= 1'b0;
			R_Valid_Other1	<= 1'b0;
			R_Valid_Other2	<= 1'b0;
			R_Nack			<= 1'b0;
		end
		else begin
			R_Valid			<= I_Valid;
			R_Valid_Other1	<= I_Valid_Other1;
			R_Valid_Other2	<= I_Valid_Other2;
			R_Nack			<= I_Nack;
		end
	end

	always_ff @( posedge clock ) begin: ff_release_aluport
		if ( reset ) begin
			R_Rls			<= 1'b0;
		end
		else begin
			R_Rls			<= I_Rls;
		end
	end


	//// Check Status												////
	assign is_FSM_Check_Attirb	= is_FSM_Check_Attirb;

	assign is_no_ready		= ~is_Ready1 & ~is_Ready2;

	assign is_ready			= is_ready;

	assign ZeroLength		= ( R_Length == '0 );

	assign AttribData		= is_FSM_Check_Attirb & is_no_ready & is_AuxData & ~I_Valid_Other1 & ~I_Valid_Other2;

	assign Wait				= ( R_FSM == wAIT_OPRAND ) & is_no_ready;


	//// Output														////
	assign O_is_Valid		= ( R_FSM != iNIT );

	assign O_SelPort		= R_Config_by_Port;

	//	 Backup Attribute Word
	assign Backup_Attrib	= is_FSM_Check_Attirb & is_no_ready & is_AuxData & R_Valid & ~R_ConfigDone;
	assign O_Backup_Attrib	= Backup_Attrib;

	//	 Synch by Nack Token
	assign O_Nack_My		= Backup_Attrib | ( Wait & ~( ( R_FSM == wAIT_OPRAND ) & is_ready & I_Valid_Other1 & I_Valid_Other2 )  );

	//	 Ready to Input
	assign O_Ready			= R_Ready;

	//	 Send Header
	assign O_SendHead		= ( ( R_FSM == hEADER ) | ( is_FSM_Check_Attirb & ~R_ConfigDone ) ) & ~is_Configured & ~( is_Ready1 | is_Ready2 );

	assign Config_Other		= is_PConfigData & R_ConfigDone & is_no_ready;

	//	 Bypassing from Input to Output

	assign O_Bypass			= R_Bypass;

	always_ff @( posedge clock ) begin
		if ( reset ) begin
			R_Bypass		<= 1'b0;
		end
		else begin
			R_Bypass		<= is_FSM_Check_Attirb & (
								is_RouteData |
								is_RConfigData |
								( is_PConfigData & R_ConfigDone ) |
								Config_Other
								) |
								( R_FSM == rOUTE ) |
								( R_FSM == rCONFIG ) |
								( R_FSM == bYPASS );
			end
		end

	//	 Input Data into DataPath
	assign O_InputData		= R_InputData | (
									( is_FSM_Check_Attirb & is_no_ready & R_Valid & ~R_ConfigDone & R_DataWord )
								) | (
									( R_FSM == wAIT_DATA ) & is_ready & (( R_Valid & R_Valid_Other1 & R_Valid_Other2 ) | R_ConfigDone )
								) | (
									( R_FSM == wAIT_OPRAND ) & ( R_DataWord & R_ConfigDone )
								);

	//	 Use of Backed up Attribute Word
	assign O_Set_Backup		= R_Set_Backup & ~R_ConfigDone;

	//	 Setting Attribute Word
	assign O_SetDAttrib		= R_SetDAttrib;

	//	 Attribute Block Length fed into Attribute Decoder
	assign O_Length			= R_Length;

	//	 Do Configuration
	assign O_Configure		= R_Configure;


	//// Capture Control and Tokens									////
	always_ff @( posedge clock ) begin: ff_config_aluport
		if ( reset ) begin
			R_Config_by_Port	<= 1'b0;
		end
		else begin
			R_Config_by_Port	<= (( R_FSM == pCONFIG ) & ~ZeroLength ) | (is_FSM_Check_Attirb & is_PConfigData & ~R_ConfigDone );
		end
	end

	always_ff @( posedge clock ) begin: ff_ready_aluport
		if ( reset ) begin
			R_Ready			<= 1'b0;
		end
		else begin
			R_Ready			<= ( R_FSM > oUT_ATTRIB ) | ( is_FSM_Check_Attirb & is_AuxData & ( I_Valid | I_Valid_Other1 | I_Valid_Other2 ) );
		end
	end

	always_ff @( posedge clock ) begin: ff_configure_aluport
		if ( reset ) begin
			R_Configure		<= 1'b0;
		end
		else begin
			R_Configure		<= (is_FSM_Check_Attirb & is_PConfigData & ~R_ConfigDone ) | ( R_FSM == pCONFIG );
		end
	end


	//// Synch on Port Control										////
	always_ff @( posedge clock ) begin: ff_fsm_aluport
		if ( reset ) begin
			R_InputData		<= 1'b0;
			R_SetPConfig	<= 1'b0;
			R_SetDAttrib	<= 1'b0;
			R_DataWord		<= 1'b0;
			R_ConfigDone	<= 1'b0;
			R_Set_Backup	<= 1'b0;
			R_Length		<= '0;
			R_FSM			<= iNIT;
		end
		else case ( R_FSM )
			iNIT: begin
				if ( I_Valid & I_Acq ) begin
					R_InputData		<= 1'b0;
					R_SetPConfig	<= 1'b0;
					R_SetDAttrib	<= 1'b0;
					R_DataWord		<= 1'b0;
					R_ConfigDone	<= 1'b0;
					R_Set_Backup	<= 1'b0;
					R_Length		<= 1;
					R_FSM			<= hEADER;
				end
				else begin
					R_InputData		<= 1'b0;
					R_SetPConfig	<= 1'b0;
					R_SetDAttrib	<= 1'b0;
					R_DataWord		<= 1'b0;
					R_ConfigDone	<= 1'b0;
					R_Set_Backup	<= 1'b0;
					R_Length		<= '0;
					R_FSM			<= iNIT;
				end
			end
			hEADER: begin
				if ( ZeroLength & I_Valid ) begin
					R_InputData		<= 1'b0;
					R_SetPConfig	<= 1'b0;
					R_SetDAttrib	<= 1'b0;
					R_DataWord		<= 1'b0;
					R_ConfigDone	<= 1'b0;
					R_Set_Backup	<= 1'b0;
					R_Length		<= '0;
					R_FSM			<= cHK_ATTRIB;
				end
				else begin
					R_InputData		<= 1'b0;
					R_SetPConfig	<= 1'b0;
					R_SetDAttrib	<= 1'b0;
					R_DataWord		<= 1'b0;
					R_ConfigDone	<= 1'b0;
					R_Set_Backup	<= 1'b0;
					R_Length		<= R_Length - I_Valid;
					R_FSM			<= hEADER;
				end
			end
			cHK_ATTRIB: begin
				if ( I_Valid ) begin
					if ( is_RouteData ) begin
						R_InputData		<= 1'b0;
						R_SetPConfig	<= 1'b0;
						R_SetDAttrib	<= 1'b0;
						R_DataWord		<= 1'b0;
						R_ConfigDone	<= R_ConfigDone;
						R_Set_Backup	<= 1'b0;
						R_Length		<= I_Length;
						R_FSM			<= rOUTE;
					end
					else if ( is_RConfigData ) begin
						R_InputData		<= 1'b0;
						R_SetPConfig	<= 1'b0;
						R_SetDAttrib	<= 1'b0;
						R_DataWord		<= 1'b0;
						R_ConfigDone	<= R_ConfigDone;
						R_Set_Backup	<= 1'b0;
						R_Length		<= I_Length;
						R_FSM			<= rCONFIG;
					end
					else if ( is_PConfigData & ~R_ConfigDone ) begin
						R_InputData		<= 1'b0;
						R_SetPConfig	<= 1'b1;
						R_SetDAttrib	<= 1'b0;
						R_DataWord		<= 1'b0;
						R_ConfigDone	<= R_ConfigDone;
						R_Set_Backup	<= 1'b0;
						R_Length		<= I_Length;
						R_FSM			<= pCONFIG;
					end
					else if ( is_PConfigData & R_ConfigDone ) begin
						R_InputData		<= 1'b0;
						R_SetPConfig	<= 1'b0;
						R_SetDAttrib	<= 1'b0;
						R_DataWord		<= 1'b0;
						R_ConfigDone	<= R_ConfigDone;
						R_Set_Backup	<= 1'b0;
						R_Length		<= I_Length;
						R_FSM			<= bYPASS;
					end
					else if ( is_AuxData & is_no_ready ) begin
						R_InputData		<= 1'b0;
						R_SetPConfig	<= 1'b0;
						R_SetDAttrib	<= 1'b0;
						R_DataWord		<= 1'b1;
						R_ConfigDone	<= R_ConfigDone;
						R_Set_Backup	<= 1'b0;
						R_Length		<= I_Length;
						R_FSM			<= wAIT_OPRAND;
					end
					else if ( I_Valid & ( is_Ready1 | is_Ready2 ) ) begin
						R_InputData		<= 1'b1;
						R_SetPConfig	<= 1'b0;
						R_SetDAttrib	<= 1'b0;
						R_DataWord		<= 1'b0;
						R_ConfigDone	<= R_ConfigDone;
						R_Set_Backup	<= 1'b0;
						R_Length		<= I_Length;
						R_FSM			<= wAIT_DATA;
					end
				end
			end
			rOUTE: begin
				if ( ZeroLength & I_Valid ) begin
					R_InputData		<= 1'b0;
					R_SetPConfig	<= 1'b0;
					R_SetDAttrib	<= 1'b0;
					R_DataWord		<= 1'b0;
					R_ConfigDone	<= R_ConfigDone;
					R_Set_Backup	<= 1'b0;
					R_Length		<= '0;
					R_FSM			<= cHK_ATTRIB;
				end
				else begin
					R_InputData		<= 1'b0;
					R_SetPConfig	<= 1'b0;
					R_SetDAttrib	<= 1'b0;
					R_DataWord		<= 1'b0;
					R_ConfigDone	<= R_ConfigDone;
					R_Set_Backup	<= 1'b0;
					R_Length		<= R_Length - I_Valid;
					R_FSM			<= rOUTE;
				end
			end
			rCONFIG: begin
				if ( ZeroLength & I_Valid ) begin
					R_InputData		<= 1'b0;
					R_SetPConfig	<= 1'b0;
					R_SetDAttrib	<= 1'b0;
					R_DataWord		<= 1'b0;
					R_ConfigDone	<= R_ConfigDone;
					R_Set_Backup	<= 1'b0;
					R_Length		<= '0;
					R_FSM			<= cHK_ATTRIB;
				end
				else begin
					R_InputData		<= 1'b0;
					R_SetPConfig	<= 1'b0;
					R_SetDAttrib	<= 1'b0;
					R_DataWord		<= 1'b0;
					R_ConfigDone	<= R_ConfigDone;
					R_Set_Backup	<= 1'b0;
					R_Length		<= R_Length - I_Valid;
					R_FSM			<= rCONFIG;
				end
			end
			pCONFIG: begin
				if ( ZeroLength & I_Valid ) begin
					R_InputData		<= 1'b0;
					R_SetPConfig	<= 1'b0;
					R_SetDAttrib	<= 1'b0;
					R_DataWord		<= 1'b0;
					R_ConfigDone	<= 1'b1;
					R_Set_Backup	<= 1'b0;
					R_Length		<= '0;
					R_FSM			<= cHK_ATTRIB;
				end
				else begin
					R_InputData		<= 1'b0;
					R_SetPConfig	<= 1'b1;
					R_SetDAttrib	<= 1'b0;
					R_DataWord		<= 1'b0;
					R_ConfigDone	<= R_ConfigDone;
					R_Set_Backup	<= 1'b0;
					R_Length		<= R_Length - I_Valid;
					R_FSM			<= pCONFIG;
				end
			end
			bYPASS: begin
				if ( ZeroLength & I_Valid ) begin
					R_InputData		<= 1'b0;
					R_SetPConfig	<= 1'b0;
					R_SetDAttrib	<= 1'b0;
					R_DataWord		<= 1'b0;
					R_ConfigDone	<= R_ConfigDone;
					R_Set_Backup	<= 1'b0;
					R_Length		<= '0;
					R_FSM			<= cHK_ATTRIB;
				end
				else begin
					R_InputData		<= 1'b0;
					R_SetPConfig	<= 1'b0;
					R_SetDAttrib	<= 1'b0;
					R_DataWord		<= 1'b0;
					R_ConfigDone	<= R_ConfigDone;
					R_Set_Backup	<= 1'b0;
					R_Length		<= R_Length - I_Valid;
					R_FSM			<= bYPASS;
				end
			end
			oUT_ATTRIB: begin
				if ( ~R_Nack ) begin
					R_InputData		<= 1'b1;
					R_SetPConfig	<= 1'b0;
					R_SetDAttrib	<= 1'b0;
					R_DataWord		<= R_DataWord;
					R_ConfigDone	<= R_ConfigDone;
					R_Set_Backup	<= 1'b0;
					R_Length		<= R_Length;
					R_FSM			<= wAIT_OPRAND;
				end
			end
			wAIT_OPRAND: begin
				if ( is_Ready1 | is_Ready2 ) begin
					R_InputData		<= 1'b0;
					R_SetPConfig	<= 1'b0;
					R_SetDAttrib	<= 1'b1;
					R_DataWord		<= R_DataWord;
					R_ConfigDone	<= R_ConfigDone;
					R_Set_Backup	<= 1'b1;
					R_Length		<= R_Length;
					R_FSM			<= wAIT_DATA;
				end
				else begin
					R_InputData		<= 1'b0;
					R_SetPConfig	<= 1'b0;
					R_SetDAttrib	<= 1'b0;
					R_DataWord		<= 1'b1;
					R_ConfigDone	<= R_ConfigDone;
					R_Set_Backup	<= 1'b0;
					R_Length		<= R_Length;
					R_FSM			<= wAIT_OPRAND;
				end
			end
			wAIT_DATA: begin
				if ( ( is_Ready1 | is_Ready2 ) & I_Valid & I_Valid_Other1 | I_Valid_Other2 ) begin
					R_InputData		<= 1'b1;
					R_SetPConfig	<= 1'b0;
					R_SetDAttrib	<= 1'b0;
					R_DataWord		<= R_DataWord;
					R_ConfigDone	<= R_ConfigDone;
					R_Set_Backup	<= 1'b0;
					R_Length		<= R_Length;
					R_FSM			<= dATA;
				end
				else begin
					R_InputData		<= 1'b1;
					R_SetPConfig	<= 1'b0;
					R_SetDAttrib	<= 1'b0;
					R_DataWord		<= R_DataWord;
					R_ConfigDone	<= R_ConfigDone;
					R_Set_Backup	<= 1'b0;
					R_Length		<= R_Length;
					R_FSM			<= wAIT_DATA;
				end
			end
			dATA: begin
				if ( R_Valid & R_Rls & I_Done ) begin
					R_InputData		<= 1'b0;
					R_SetPConfig	<= 1'b0;
					R_SetDAttrib	<= 1'b0;
					R_DataWord		<= 1'b0;
					R_ConfigDone	<= R_ConfigDone;
					R_Set_Backup	<= 1'b0;
					R_Length		<= '0;
					R_FSM			<= iNIT;
				end
				else if ( R_Valid & R_Rls ) begin
					R_InputData		<= 1'b1;
					R_SetPConfig	<= 1'b0;
					R_SetDAttrib	<= 1'b0;
					R_DataWord		<= R_DataWord;
					R_ConfigDone	<= R_ConfigDone;
					R_Set_Backup	<= 1'b0;
					R_Length		<= R_Length;
					R_FSM			<= eND_SEQ;
				end
				else begin
					R_InputData		<= 1'b1;
					R_SetPConfig	<= 1'b0;
					R_SetDAttrib	<= 1'b0;
					R_DataWord		<= R_DataWord;
					R_ConfigDone	<= R_ConfigDone;
					R_Set_Backup	<= 1'b0;
					R_Length		<= R_Length;
					R_FSM			<= dATA;
				end
			end
			eND_SEQ: begin
				if ( I_Done ) begin
					R_InputData		<= 1'b0;
					R_SetPConfig	<= 1'b0;
					R_SetDAttrib	<= 1'b0;
					R_DataWord		<= 1'b0;
					R_ConfigDone	<= 1'b0;
					R_Length		<= '0;
					R_FSM			<= iNIT;
				end
			end
			default: begin
				R_InputData		<= 1'b0;
				R_SetPConfig	<= 1'b0;
				R_SetDAttrib	<= 1'b0;
				R_DataWord		<= 1'b0;
				R_ConfigDone	<= 1'b0;
				R_Length		<= '0;
				R_FSM			<= iNIT;
			end
		endcase
	end

endmodule
