//******************************************************************************
//
// File:    Opcode.java
// Package: edu.rit.pj2.tracker
// Unit:    Class edu.rit.pj2.tracker.Opcode
//
// This Java source file is copyright (C) 2013 by Alan Kaminsky. All rights
// reserved. For further information, contact the author, Alan Kaminsky, at
// ark@cs.rit.edu.
//
// This Java source file is part of the Parallel Java 2 Library ("PJ2"). PJ2 is
// free software; you can redistribute it and/or modify it under the terms of
// the GNU General Public License as published by the Free Software Foundation;
// either version 3 of the License, or (at your option) any later version.
//
// PJ2 is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
// A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// A copy of the GNU General Public License is provided in the file gpl.txt. You
// may also obtain a copy of the GNU General Public License on the World Wide
// Web at http://www.gnu.org/licenses/gpl.html.
//
//******************************************************************************

package edu.rit.pj2.tracker;

/**
 * Class Opcode provides opcodes for the PJ2 binary protocol.
 *
 * @author  Alan Kaminsky
 * @version 10-Jul-2013
 */
class Opcode
	{
	// Prevent construction.
	private Opcode()
		{
		}

	// Opcodes denoting type of entity at the far end.
	public static final byte JOB                                = (byte) 1;
	public static final byte TRACKER                            = (byte) 2;
	public static final byte LAUNCHER                           = (byte) 3;
	public static final byte BACKEND                            = (byte) 4;

	// Opcodes denoting methods in interface JobRef.
	public static final byte JOBREF_JOB_LAUNCHED                = (byte) 5;
	public static final byte JOBREF_JOB_STARTED                 = (byte) 6;
	public static final byte JOBREF_TASK_LAUNCHING              = (byte) 7;
	public static final byte JOBREF_TASK_LAUNCHED               = (byte) 8;
	public static final byte JOBREF_TAKE_TUPLE                  = (byte) 9;
	public static final byte JOBREF_WRITE_TUPLE                 = (byte) 10;
	public static final byte JOBREF_TASK_FINISHED               = (byte) 11;
	public static final byte JOBREF_TASK_FAILED                 = (byte) 12;
	public static final byte JOBREF_HEARTBEAT_FROM_TRACKER      = (byte) 13;
	public static final byte JOBREF_HEARTBEAT_FROM_TASK         = (byte) 14;
	public static final byte JOBREF_WRITE_STANDARD_STREAM       = (byte) 15;

	// Opcodes denoting methods in interface TrackerRef.
	public static final byte TRACKERREF_LAUNCHER_STARTED        = (byte) 16;
	public static final byte TRACKERREF_LAUNCHER_STOPPED        = (byte) 17;
	public static final byte TRACKERREF_LAUNCH_JOB              = (byte) 18;
	public static final byte TRACKERREF_LAUNCH_TASK_GROUP       = (byte) 19;
	public static final byte TRACKERREF_LAUNCH_FAILED           = (byte) 20;
	public static final byte TRACKERREF_TASK_DONE               = (byte) 21;
	public static final byte TRACKERREF_JOB_DONE                = (byte) 22;
	public static final byte TRACKERREF_STOP_JOB                = (byte) 23;
	public static final byte TRACKERREF_HEARTBEAT_FROM_JOB      = (byte) 24;
	public static final byte TRACKERREF_HEARTBEAT_FROM_LAUNCHER = (byte) 25;

	// Opcodes denoting methods in interface LauncherRef.
	public static final byte LAUNCHERREF_LAUNCH                 = (byte) 26;
	public static final byte LAUNCHERREF_HEARTBEAT_FROM_TRACKER = (byte) 27;

	// Opcodes denoting methods in interface BackendRef.
	public static final byte BACKENDREF_START_TASK              = (byte) 28;
	public static final byte BACKENDREF_TUPLE_TAKEN             = (byte) 29;
	public static final byte BACKENDREF_STOP_TASK               = (byte) 30;
	public static final byte BACKENDREF_HEARTBEAT_FROM_JOB      = (byte) 31;

	// Opcode for shutting down the connection.
	public static final byte SHUTDOWN                           = (byte) 255;
	}
