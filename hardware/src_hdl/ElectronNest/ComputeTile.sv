///////////////////////////////////////////////////////////////////////////////////////////////////
//
//	ElectronNest
//	Copyright (C) 2024  Shigeyuki TAKANO
//
//  GNU AFFERO GENERAL PUBLIC LICENSE
//	version 3.0
//
//	Compute Tile
//		Module Name:	ComputeTile
//		Function:
//						Top Module of Grid Array
//						Two Dimensional Array consistting of PE and RE
//						Current Topology;
//							- Two Dimensional Mesh
//							- Edge Element connects its output to its input
//
///////////////////////////////////////////////////////////////////////////////////////////////////

module ComputeTile
	import	pkg_en::*;
#(
	parameter int EnIF_North		= 1,
	parameter int EnIF_East			= 1,
	parameter int EnIF_West			= 1,
	parameter int EnIF_South		= 1,
	parameter int WIDTH_DATA		= 32,
	parameter int NUM_LINK			= 4,
	parameter int NUM_IF			= EnIF_North+EnIF_East+EnIF_West+EnIF_South,
	parameter int NUM_CHANNEL		= 2,
	parameter int WIDTH_LENGTH		= 10,
	parameter int WIDTH_ADDR		= 8,
	parameter int WIDTH_UNIT		= 8,
	parameter int NUM_MEMUNIT		= 4,
	parameter int SIZE_CRAM			= 256,
	parameter int DEPTH_FIFO		= 16,
	parameter int WIDTH_OPCODE		= 8,
	parameter int WIDTH_CONSTANT	= 8,
	parameter int NUM_WORKER		= 4,
	parameter int NUM_ROW			= 4,
	parameter int NUM_CLM			= 4,
	parameter int NUM_ALU			= 2,
	parameter int NUM_CRAM			= 2,
	parameter int ExtdConfig		= 0
)(
	input							clock,
	input							reset,

	input	FTk_cclm_if_t			I_FTk,							//Input Forward-Tokens
	output	BTk_cclm_if_t			O_BTk,							//Output Backward-Tokens
	output	FTk_cclm_if_t			O_FTk,							//Output Forward-Tokens
	input	BTk_cclm_if_t			I_BTk							//Input Backward-Tokens
);


	//// Port ID Assignment											////
	localparam int NorthIF_ID	= 0;
	localparam int EastIF_ID	= EnIF_North;
	localparam int WestIF_ID	= EnIF_North + EnIF_East;
	localparam int SouthIF_ID	= EnIF_North + EnIF_East + EnIF_West;


	//// RE Path													////
	FTk_clclmrow_t				R_I_FTk;
	BTk_clclmrow_t				R_O_BTk;
	FTk_clclmrow_t				R_O_FTk;
	BTk_clclmrow_t				R_I_BTk;


	//// PE Path													////
	FTk_clclmrow_t				P_I_FTk;
	BTk_clclmrow_t				P_O_BTk;
	FTk_clclmrow_t				P_O_FTk;
	BTk_clclmrow_t				P_I_BTk;

	bit2_clclmrow_t				R_I_InC;
	bit2_clclmrow_t				R_O_InC;
	bit2_clclmrow_t				P_I_InC;
	bit2_clclmrow_t				P_O_InC;


	for ( genvar r = 0; r < NUM_ROW; ++r ) begin : g_row
		for ( genvar c = 0; c < (NUM_CLM+1)/2; ++c ) begin : g_clm
			RE #(
				.WIDTH_DATA(	WIDTH_DATA				),
				.NUM_CRAM(		NUM_CRAM				),
				.NUM_LINK(		NUM_LINK				),
				.NUM_CHANNEL(	NUM_CHANNEL				),
				.WIDTH_ADDR(	WIDTH_ADDR				),
				.WIDTH_LENGTH(	WIDTH_LENGTH			),
				.DEPTH_FIFO(	DEPTH_FIFO				),
				.WIDTH_UNIT(	WIDTH_UNIT				),
				.NUM_MEMUNIT(	NUM_MEMUNIT				),
				.SIZE_CRAM(		SIZE_CRAM				),
				.ExtdConfig (	ExtdConfig				)
			) RE (
				.clock(			clock					),
				.reset(			reset					),
				.I_FTk(			R_I_FTk[ r ][ c ]		),
				.O_BTk(			R_O_BTk[ r ][ c ]		),
				.O_FTk(			R_O_FTk[ r ][ c ]		),
				.I_BTk(			R_I_BTk[ r ][ c ]		),
				.I_InC(			R_I_InC[ r ][ c ]		),
				.O_InC(			R_O_InC[ r ][ c ]		)
			);

			PE #(
				.WIDTH_DATA(	WIDTH_DATA				),
				.NUM_ALU(		NUM_ALU					),
				.NUM_LINK(		NUM_LINK				),
				.NUM_CHANNEL(	NUM_CHANNEL				),
				.WIDTH_LENGTH(	WIDTH_LENGTH			),
				.DEPTH_FIFO(	DEPTH_FIFO				),
				.WIDTH_OPCODE(	WIDTH_OPCODE			),
				.WIDTH_CONSTANT(WIDTH_CONSTANT			),
				.NUM_WORKER(	NUM_WORKER				)
			) PE (
				.clock(			clock					),
				.reset(			reset					),
				.I_FTk(			P_I_FTk[ r ][ c ]		),
				.O_BTk(			P_O_BTk[ r ][ c ]		),
				.O_FTk(			P_O_FTk[ r ][ c ]		),
				.I_BTk(			P_I_BTk[ r ][ c ]		),
				.I_InC(			P_I_InC[ r ][ c ]		),
				.O_InC(			P_O_InC[ r ][ c ]		)
			);
		end
	end


	//// PE <-> RE Connection										////
	for ( genvar	k = 0; k < NUM_CHANNEL; ++k ) begin: g_pe_re
		for ( genvar r = 0; r < NUM_ROW; ++r ) begin: g_pe_re_row
			for ( genvar c = 0; c < NUM_CLM; ++c ) begin: g_pre_re_clm

				if ((c%2 == 0) & (r%2 == 0))	begin
					if (r > 0) begin
						//PE to RE: South-Port to North-Port
						assign R_I_FTk[ r ][ c/2 ][0][ k ]	= P_O_FTk[r-1][ c/2 ][3][ k ];
						assign R_I_BTk[ r ][ c/2 ][0][ k ]	= P_O_BTk[r-1][ c/2 ][3][ k ];
						assign R_I_InC[ r ][ c/2 ][0][ k ]	= P_O_InC[r-1][ c/2 ][3][ k ];

						//RE to PE: North-Port to South-Port
						assign P_I_FTk[r-1][ c/2 ][3][ k ]	= R_O_FTk[ r ][ c/2 ][0][ k ];
						assign P_I_BTk[r-1][ c/2 ][3][ k ]	= R_O_BTk[ r ][ c/2 ][0][ k ];
						assign P_I_InC[r-1][ c/2 ][3][ k ]	= R_O_InC[ r ][ c/2 ][0][ k ];
					end
					if (r < (NUM_ROW-1)) begin
						//PE to RE: North-Port to South-Port
						assign R_I_FTk[ r ][ c/2 ][3][ k ]		= P_O_FTk[ r+1 ][ c/2 ][0][ k ];
						assign R_I_BTk[ r ][ c/2 ][3][ k ]		= P_O_BTk[ r+1 ][ c/2 ][0][ k ];
						assign R_I_InC[ r ][ c/2 ][3][ k ]		= P_O_InC[ r+1 ][ c/2 ][0][ k ];

						//RE to PE: South-Port to North-Port
						assign P_I_FTk[ r+1 ][ c/2 ][0][ k ]	= R_O_FTk[ r ][ c/2 ][3][ k ];
						assign P_I_BTk[ r+1 ][ c/2 ][0][ k ]	= R_O_BTk[ r ][ c/2 ][3][ k ];
						assign P_I_InC[ r+1 ][ c/2 ][0][ k ]	= R_O_InC[ r ][ c/2 ][3][ k ];
					end

					if (c > 1) begin
						//PE to RE: East-Port to West-Port
						assign R_I_FTk[ r ][ c/2 ][2][ k ]		= P_O_FTk[ r ][ c/2-1 ][1][ k ];
						assign R_I_BTk[ r ][ c/2 ][2][ k ]		= P_O_BTk[ r ][ c/2-1 ][1][ k ];
						assign R_I_InC[ r ][ c/2 ][2][ k ]		= P_O_InC[ r ][ c/2-1 ][1][ k ];

						//RE to PE: West-Port to East-Port
						assign P_I_FTk[ r ][ c/2-1 ][1][ k ]	= R_O_FTk[ r ][ c/2 ][2][ k ];
						assign P_I_BTk[ r ][ c/2-1 ][1][ k ]	= R_O_BTk[ r ][ c/2 ][2][ k ];
						assign P_I_InC[ r ][ c/2-1 ][1][ k ]	= R_O_InC[ r ][ c/2 ][2][ k ];
					end
					if (c < (NUM_CLM-1)) begin
						//PE to RE: West-Port to East-Port
						assign R_I_FTk[ r ][ c/2 ][1][ k ]	= P_O_FTk[ r ][ c/2 ][2][ k ];
						assign R_I_BTk[ r ][ c/2 ][1][ k ]	= P_O_BTk[ r ][ c/2 ][2][ k ];
						assign R_I_InC[ r ][ c/2 ][1][ k ]	= P_O_InC[ r ][ c/2 ][2][ k ];

						//RE to PE: East-Port to West-Port
						assign P_I_FTk[ r ][ c/2 ][2][ k ]	= R_O_FTk[ r ][ c/2 ][1][ k ];
						assign P_I_BTk[ r ][ c/2 ][2][ k ]	= R_O_BTk[ r ][ c/2 ][1][ k ];
						assign P_I_InC[ r ][ c/2 ][2][ k ]	= R_O_InC[ r ][ c/2 ][1][ k ];
					end
				end
				else if ((c%2 == 0) & (r%2 == 1))	begin
					if (r < (NUM_ROW-1)) begin
						//PE to RE: North-Port to South-Port
						assign P_I_FTk[ r ][ c/2 ][3][ k ]		= R_O_FTk[ r+1 ][ c/2 ][0][ k ];
						assign P_I_BTk[ r ][ c/2 ][3][ k ]		= R_O_BTk[ r+1 ][ c/2 ][0][ k ];
						assign P_I_InC[ r ][ c/2 ][3][ k ]		= R_O_InC[ r+1 ][ c/2 ][0][ k ];

						//RE to PE: South-Port to North-Port
						assign R_I_FTk[ r+1 ][ c/2 ][0][ k ]	= P_O_FTk[ r ][ c/2 ][3][ k ];
						assign R_I_BTk[ r+1 ][ c/2 ][0][ k ]	= P_O_BTk[ r ][ c/2 ][3][ k ];
						assign R_I_InC[ r+1 ][ c/2 ][0][ k ]	= P_O_InC[ r ][ c/2 ][3][ k ];
					end
					//PE to RE: South-Port to North-Port
					assign R_I_FTk[r-1][ c/2 ][3][ k ]	= P_O_FTk[ r ][ c/2 ][0][ k ];
					assign R_I_BTk[r-1][ c/2 ][3][ k ]	= P_O_BTk[ r ][ c/2 ][0][ k ];
					assign R_I_InC[r-1][ c/2 ][3][ k ]	= P_O_InC[ r ][ c/2 ][0][ k ];

					//RE to PE: North-Port to South-Port
					assign P_I_FTk[ r ][ c/2 ][0][ k ]	= R_O_FTk[r-1][ c/2 ][3][ k ];
					assign P_I_BTk[ r ][ c/2 ][0][ k ]	= R_O_BTk[r-1][ c/2 ][3][ k ];
					assign P_I_InC[ r ][ c/2 ][0][ k ]	= R_O_InC[r-1][ c/2 ][3][ k ];

					if (c > 1) begin
						//PE to RE: East-Port to West-Port
						assign P_I_FTk[ r ][ c/2 ][2][ k ]		= R_O_FTk[ r ][ c/2-1 ][1][ k ];
						assign P_I_BTk[ r ][ c/2 ][2][ k ]		= R_O_BTk[ r ][ c/2-1 ][1][ k ];
						assign P_I_InC[ r ][ c/2 ][2][ k ]		= R_O_InC[ r ][ c/2-1 ][1][ k ];

						//RE to PE: West-Port to East-Port
						assign R_I_FTk[ r ][ c/2-1 ][1][ k ]	= P_O_FTk[ r ][ c/2 ][2][ k ];
						assign R_I_BTk[ r ][ c/2-1 ][1][ k ]	= P_O_BTk[ r ][ c/2 ][2][ k ];
						assign R_I_InC[ r ][ c/2-1 ][1][ k ]	= P_O_InC[ r ][ c/2 ][2][ k ];
					end
					if (c < (NUM_CLM-1)) begin
						//PE to RE: East-Port to West-Port
						assign R_I_FTk[ r ][ c/2 ][2][ k ]	= P_O_FTk[ r ][ c/2 ][1][ k ];
						assign R_I_BTk[ r ][ c/2 ][2][ k ]	= P_O_BTk[ r ][ c/2 ][1][ k ];
						assign R_I_InC[ r ][ c/2 ][2][ k ]	= P_O_InC[ r ][ c/2 ][1][ k ];

						//RE to PE: West-Port to East-Port
						assign P_I_FTk[ r ][ c/2 ][1][ k ]	= R_O_FTk[ r ][ c/2 ][2][ k ];
						assign P_I_BTk[ r ][ c/2 ][1][ k ]	= R_O_BTk[ r ][ c/2 ][2][ k ];
						assign P_I_InC[ r ][ c/2 ][1][ k ]	= R_O_InC[ r ][ c/2 ][2][ k ];
					end
				end
				else if ((c%2 == 1) & (r%2 == 0))	begin
					if (r > 0) begin
						//RE to PE: South-Port to North-Port
						assign P_I_FTk[ r ][ c/2 ][0][ k ]	= R_O_FTk[r-1][ c/2 ][3][ k ];
						assign P_I_BTk[ r ][ c/2 ][0][ k ]	= R_O_BTk[r-1][ c/2 ][3][ k ];
						assign P_I_InC[ r ][ c/2 ][0][ k ]	= R_O_InC[r-1][ c/2 ][3][ k ];

						//PE to RE: North-Port to South-Port
						assign R_I_FTk[r-1][ c/2 ][3][ k ]	= P_O_FTk[ r ][ c/2 ][0][ k ];
						assign R_I_BTk[r-1][ c/2 ][3][ k ]	= P_O_BTk[ r ][ c/2 ][0][ k ];
						assign R_I_InC[r-1][ c/2 ][3][ k ]	= P_O_InC[ r ][ c/2 ][0][ k ];
					end
					//PE to RE: South-Port to North-Port
					assign R_I_FTk[ r+1 ][ c/2 ][0][ k ]	= P_O_FTk[ r ][ c/2 ][3][ k ];
					assign R_I_BTk[ r+1 ][ c/2 ][0][ k ]	= P_O_BTk[ r ][ c/2 ][3][ k ];
					assign R_I_InC[ r+1 ][ c/2 ][0][ k ]	= P_O_InC[ r ][ c/2 ][3][ k ];

					//RE to PE: North-Port to South-Port
					assign P_I_FTk[ r ][ c/2 ][3][ k ]		= R_O_FTk[ r+1 ][ c/2 ][0][ k ];
					assign P_I_BTk[ r ][ c/2 ][3][ k ]		= R_O_BTk[ r+1 ][ c/2 ][0][ k ];
					assign P_I_InC[ r ][ c/2 ][3][ k ]		= R_O_InC[ r+1 ][ c/2 ][0][ k ];

					if (c < (NUM_CLM-1)) begin
						//RE to PE: West-Port to East-Port
						assign P_I_FTk[ r ][ c/2 ][1][ k ]		= R_O_FTk[ r ][ c/2+1 ][2][ k ];
						assign P_I_BTk[ r ][ c/2 ][1][ k ]		= R_O_BTk[ r ][ c/2+1 ][2][ k ];
						assign P_I_InC[ r ][ c/2 ][1][ k ]		= R_O_InC[ r ][ c/2+1 ][2][ k ];

						//RE to PE: West-Port to East-Port
						assign R_I_FTk[ r ][ c/2+1 ][2][ k ]	= P_O_FTk[ r ][ c/2 ][1][ k ];
						assign R_I_BTk[ r ][ c/2+1 ][2][ k ]	= P_O_BTk[ r ][ c/2 ][1][ k ];
						assign R_I_InC[ r ][ c/2+1 ][2][ k ]	= P_O_InC[ r ][ c/2 ][1][ k ];
					end
					//RE to PE: West-Port to East-Port
					assign R_I_FTk[ r ][ c/2 ][1][ k ]	= P_O_FTk[ r ][ c/2 ][2][ k ];
					assign R_I_BTk[ r ][ c/2 ][1][ k ]	= P_O_BTk[ r ][ c/2 ][2][ k ];
					assign R_I_InC[ r ][ c/2 ][1][ k ]	= P_O_InC[ r ][ c/2 ][2][ k ];

					//PE to RE: East-Port to West-Port
					assign P_I_FTk[ r ][ c/2 ][2][ k ]	= R_O_FTk[ r ][ c/2 ][1][ k ];
					assign P_I_BTk[ r ][ c/2 ][2][ k ]	= R_O_BTk[ r ][ c/2 ][1][ k ];
					assign P_I_InC[ r ][ c/2 ][2][ k ]	= R_O_InC[ r ][ c/2 ][1][ k ];
				end
				else if ((c%2 == 1) & (r%2 == 1))	begin
					if (r < (NUM_ROW-1)) begin
						//PE to RE: North-Port to South-Port
						assign R_I_FTk[ r ][ c/2 ][3][ k ]		= P_O_FTk[ r+1 ][ c/2 ][0][ k ];
						assign R_I_BTk[ r ][ c/2 ][3][ k ]		= P_O_BTk[ r+1 ][ c/2 ][0][ k ];
						assign R_I_InC[ r ][ c/2 ][3][ k ]		= P_O_InC[ r+1 ][ c/2 ][0][ k ];

						//RE to PE: South-Port to North-Port
						assign P_I_FTk[ r+1 ][ c/2 ][0][ k ]	= R_O_FTk[ r ][ c/2 ][3][ k ];
						assign P_I_BTk[ r+1 ][ c/2 ][0][ k ]	= R_O_BTk[ r ][ c/2 ][3][ k ];
						assign P_I_InC[ r+1 ][ c/2 ][0][ k ]	= R_O_InC[ r ][ c/2 ][3][ k ];
					end
					//RE to PE: North-Port to South-Port
					assign P_I_FTk[r-1][ c/2 ][3][ k ]	= R_O_FTk[ r ][ c/2 ][0][ k ];
					assign P_I_BTk[r-1][ c/2 ][3][ k ]	= R_O_BTk[ r ][ c/2 ][0][ k ];
					assign P_I_InC[r-1][ c/2 ][3][ k ]	= R_O_InC[ r ][ c/2 ][0][ k ];

					//PE to RE: South-Port to North-Port
					assign R_I_FTk[ r ][ c/2 ][0][ k ]	= P_O_FTk[r-1][ c/2 ][3][ k ];
					assign R_I_BTk[ r ][ c/2 ][0][ k ]	= P_O_BTk[r-1][ c/2 ][3][ k ];
					assign R_I_InC[ r ][ c/2 ][0][ k ]	= P_O_InC[r-1][ c/2 ][3][ k ];

					if (c < (NUM_CLM-1)) begin
						//RE to PE: West-Port to East-Port
						assign R_I_FTk[ r ][ c/2 ][1][ k ]		= P_O_FTk[ r ][ c/2+1 ][2][ k ];
						assign R_I_BTk[ r ][ c/2 ][1][ k ]		= P_O_BTk[ r ][ c/2+1 ][2][ k ];
						assign R_I_InC[ r ][ c/2 ][1][ k ]		= P_O_InC[ r ][ c/2+1 ][2][ k ];

						//PE to RE: East-Port to West-Port
						assign P_I_FTk[ r ][ c/2+1 ][2][ k ]	= R_O_FTk[ r ][ c/2 ][1][ k ];
						assign P_I_BTk[ r ][ c/2+1 ][2][ k ]	= R_O_BTk[ r ][ c/2 ][1][ k ];
						assign P_I_InC[ r ][ c/2+1 ][2][ k ]	= R_O_InC[ r ][ c/2 ][1][ k ];
					end
					//RE to PE: West-Port to East-Port
					assign P_I_FTk[ r ][ c/2 ][1][ k ]	= R_O_FTk[ r ][ c/2 ][2][ k ];
					assign P_I_BTk[ r ][ c/2 ][1][ k ]	= R_O_BTk[ r ][ c/2 ][2][ k ];
					assign P_I_InC[ r ][ c/2 ][1][ k ]	= R_O_InC[ r ][ c/2 ][2][ k ];

					//PE to RE: East-Port to West-Port
					assign R_I_FTk[ r ][ c/2 ][2][ k ]	= P_O_FTk[ r ][ c/2 ][1][ k ];
					assign R_I_BTk[ r ][ c/2 ][2][ k ]	= P_O_BTk[ r ][ c/2 ][1][ k ];
					assign R_I_InC[ r ][ c/2 ][2][ k ]	= P_O_InC[ r ][ c/2 ][1][ k ];
				end


				// Upside-Boundary
				if ( EnIF_North == 1 ) begin
					if (r == 0) begin
						if (c%2 == 1) begin
							assign P_I_FTk[0][ c/2 ][0][ k ]	= I_FTk[NorthIF_ID][ c ][ k ];
							assign P_I_BTk[0][ c/2 ][0][ k ]	= I_BTk[NorthIF_ID][ c ][ k ];
							assign P_I_InC[0][ c/2 ][0][ k ]	= '0;
						end
						else begin
							assign R_I_FTk[0][ c/2 ][0][ k ]	= I_FTk[NorthIF_ID][ c ][ k ];
							assign R_I_BTk[0][ c/2 ][0][ k ]	= I_BTk[NorthIF_ID][ c ][ k ];
							assign R_I_InC[0][ c/2 ][0][ k ]	= '0;
						end
					end
				end
				else begin
					if (r == 0) begin
						if (c%2 == 1) begin
							assign P_I_FTk[0][ c/2 ][0][ k ]	= P_O_FTk[0][ c/2 ][0][ k ];
							assign P_I_BTk[0][ c/2 ][0][ k ]	= P_O_BTk[0][ c/2 ][0][ k ];
							assign P_I_InC[0][ c/2 ][0][ k ]	= P_O_InC[0][ c/2 ][0][ k ];
						end
						else begin
							assign R_I_FTk[0][ c/2 ][0][ k ]	= R_O_FTk[0][ c/2 ][0][ k ];
							assign R_I_BTk[0][ c/2 ][0][ k ]	= R_O_FTk[0][ c/2 ][0][ k ];
							assign R_I_InC[0][ c/2 ][0][ k ]	= R_O_InC[0][ c/2 ][0][ k ];
						end
					end
				end

				// Left-side Boundary
				if ( EnIF_West == 1 ) begin
					if (c == 0) begin
						if (r%2 == 1) begin
							assign P_I_FTk[ r ][0][2][ k ]	= I_FTk[WestIF_ID][ r ][ k ];
							assign P_I_BTk[ r ][0][2][ k ]	= I_BTk[WestIF_ID][ r ][ k ];
							assign P_I_InC[ r ][0][2][ k ]	= '0;
						end
						else begin
							assign R_I_FTk[ r ][0][2][ k ]	= I_FTk[WestIF_ID][ r ][ k ];
							assign R_I_BTk[ r ][0][2][ k ]	= I_BTk[WestIF_ID][ r ][ k ];
							assign R_I_InC[ r ][0][2][ k ]	= '0;
						end
					end
				end
				else begin
					if (c == 0) begin
						if (r%2 == 1) begin
							assign P_I_FTk[ r ][0][2][ k ]	= P_O_FTk[ r ][0][2][ k ];
							assign P_I_BTk[ r ][0][2][ k ]	= P_O_BTk[ r ][0][2][ k ];
							assign P_I_InC[ r ][0][2][ k ]	= P_O_InC[ r ][0][2][ k ];
						end
						else begin
							assign R_I_FTk[ r ][0][2][ k ]	= R_O_FTk[ r ][0][2][ k ];
							assign R_I_BTk[ r ][0][2][ k ]	= R_O_BTk[ r ][0][2][ k ];
							assign R_I_InC[ r ][0][2][ k ]	= R_O_InC[ r ][0][2][ k ];
						end
					end
				end

				// Bottom-side Boundary
				if ( EnIF_South == 1 ) begin
					if (r == (NUM_ROW-1)) begin
						if (c%2 == 0) begin
							assign P_I_FTk[NUM_ROW-1][ c/2 ][3][ k ]	= I_FTk[SouthIF_ID][ c ][ k ];
							assign P_I_BTk[NUM_ROW-1][ c/2 ][3][ k ]	= I_BTk[SouthIF_ID][ c ][ k ];
							assign P_I_InC[NUM_ROW-1][ c/2 ][3][ k ]	= '0;
						end
						else begin
							assign R_I_FTk[NUM_ROW-1][ c/2 ][3][ k ]	= I_FTk[SouthIF_ID][ c ][ k ];
							assign R_I_BTk[NUM_ROW-1][ c/2 ][3][ k ]	= I_BTk[SouthIF_ID][ c ][ k ];
							assign R_I_InC[NUM_ROW-1][ c/2 ][3][ k ]	= '0;
						end
					end
				end
				else begin
					if (r == (NUM_ROW-1)) begin
						if (c%2 == 0) begin
							assign P_I_FTk[NUM_ROW-1][ c/2 ][3][ k ]	= P_O_FTk[NUM_ROW-1][ c/2 ][3][ k ];
							assign P_I_BTk[NUM_ROW-1][ c/2 ][3][ k ]	= P_O_BTk[NUM_ROW-1][ c/2 ][3][ k ];
							assign P_I_InC[NUM_ROW-1][ c/2 ][3][ k ]	= P_O_InC[NUM_ROW-1][ c/2 ][3][ k ];
						end
						else begin
							assign R_I_FTk[NUM_ROW-1][ c/2 ][3][ k ]	= R_O_FTk[NUM_ROW-1][ c/2 ][3][ k ];
							assign R_I_BTk[NUM_ROW-1][ c/2 ][3][ k ]	= R_O_BTk[NUM_ROW-1][ c/2 ][3][ k ];
							assign R_I_InC[NUM_ROW-1][ c/2 ][3][ k ]	= R_O_InC[NUM_ROW-1][ c/2 ][3][ k ];
						end
					end
				end

				// Right-side Boundary
				if ( EnIF_East == 1 ) begin
					if (c == (NUM_CLM-1)) begin
						if (r%2 == 0) begin
							assign P_I_FTk[ r ][(NUM_CLM-1)/2][1][ k ]	= I_FTk[EastIF_ID][ r ][ k ];
							assign P_I_BTk[ r ][(NUM_CLM-1)/2][1][ k ]	= I_BTk[EastIF_ID][ r ][ k ];
							assign P_I_InC[ r ][(NUM_CLM-1)/2][1][ k ]	= '0;
						end
						else begin
							assign R_I_FTk[ r ][(NUM_CLM-1)/2][1][ k ]	= I_FTk[EastIF_ID][ r ][ k ];
							assign R_I_BTk[ r ][(NUM_CLM-1)/2][1][ k ]	= I_BTk[EastIF_ID][ r ][ k ];
							assign R_I_InC[ r ][(NUM_CLM-1)/2][1][ k ]	= '0;
						end
					end
				end
				else begin
					if (c == (NUM_CLM-1)) begin
						if (r%2 == 0) begin
							assign P_I_FTk[ r ][(NUM_CLM-1)/2][1][ k ]	= P_O_FTk[ r ][(NUM_CLM-1)/2][1][ k ];
							assign P_I_BTk[ r ][(NUM_CLM-1)/2][1][ k ]	= P_O_BTk[ r ][(NUM_CLM-1)/2][1][ k ];
							assign P_I_InC[ r ][(NUM_CLM-1)/2][1][ k ]	= P_O_InC[ r ][(NUM_CLM-1)/2][1][ k ];
						end
						else begin
							assign R_I_FTk[ r ][(NUM_CLM-1)/2][1][ k ]	= R_O_FTk[ r ][(NUM_CLM-1)/2][1][ k ];
							assign R_I_BTk[ r ][(NUM_CLM-1)/2][1][ k ]	= R_O_BTk[ r ][(NUM_CLM-1)/2][1][ k ];
							assign R_I_InC[ r ][(NUM_CLM-1)/2][1][ k ]	= R_O_InC[ r ][(NUM_CLM-1)/2][1][ k ];
						end
					end
				end
			end
		end
	end


	for ( genvar k = 0; k < NUM_CHANNEL; ++k ) begin: g_port
		for ( genvar r = 0; r < NUM_ROW; ++r ) begin: g_port_row
			for ( genvar c = 0; c < NUM_CLM; ++c ) begin: g_port_clm

				// Upside-Boundary
				if ( EnIF_North == 1 ) begin
					if (r == 0) begin
						if (c%2 == 1) begin
							assign O_FTk[NorthIF_ID][ c ][ k ]	= P_O_FTk[0][ c/2 ][0][ k ];
							assign O_BTk[NorthIF_ID][ c ][ k ]	= P_O_BTk[0][ c/2 ][0][ k ];
						end
						else begin
							assign O_FTk[NorthIF_ID][ c ][ k ]	= R_O_FTk[0][ c/2 ][0][ k ];
							assign O_BTk[NorthIF_ID][ c ][ k ]	= R_O_BTk[0][ c/2 ][0][ k ];
						end
					end
				end

				// Left-side Boundary
				if ( EnIF_West == 1 ) begin
					if (c == 0) begin
						if (r%2 == 1) begin
							assign O_FTk[WestIF_ID][ r ][ k ]	= P_O_FTk[ r ][0][2][ k ];
							assign O_BTk[WestIF_ID][ r ][ k ]	= P_O_BTk[ r ][0][2][ k ];
						end
						else begin
							assign O_FTk[WestIF_ID][ r ][ k ]	= R_O_FTk[ r ][0][2][ k ];
							assign O_BTk[WestIF_ID][ r ][ k ]	= R_O_BTk[ r ][0][2][ k ];
						end
					end
				end

				// Bottom-side Boundary
				if ( EnIF_South == 1 ) begin
					if (r == (NUM_ROW-1)) begin
						if (c%2 == 0) begin
							assign O_FTk[SouthIF_ID][ c ][ k ]	= P_O_FTk[NUM_ROW-1][ c/2 ][3][ k ];
							assign O_BTk[SouthIF_ID][ c ][ k ]	= P_O_BTk[NUM_ROW-1][ c/2 ][3][ k ];
						end
						else begin
							assign O_FTk[SouthIF_ID][ c ][ k ]	= R_O_FTk[NUM_ROW-1][ c/2 ][3][ k ];
							assign O_BTk[SouthIF_ID][ c ][ k ]	= R_O_BTk[NUM_ROW-1][ c/2 ][3][ k ];
						end
					end
				end

				// Right-side Boundary
				if ( EnIF_East == 1 ) begin
					if (c == (NUM_CLM-1)) begin
						if (r%2 == 0) begin
							assign O_FTk[EastIF_ID][ r ][ k ]	= P_O_FTk[ r ][(NUM_CLM-1)/2][1][ k ];
							assign O_BTk[EastIF_ID][ r ][ k ]	= P_O_BTk[ r ][(NUM_CLM-1)/2][1][ k ];
						end
						else begin
							assign O_FTk[EastIF_ID][ r ][ k ]	= R_O_FTk[ r ][(NUM_CLM-1)/2][1][ k ];
							assign O_BTk[EastIF_ID][ r ][ k ]	= R_O_BTk[ r ][(NUM_CLM-1)/2][1][ k ];
						end
					end
				end
			end
		end
	end

endmodule
