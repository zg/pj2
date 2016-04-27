//******************************************************************************
//
// File:    Receiver.java
// Package: edu.rit.pj2.tracker
// Unit:    Class edu.rit.pj2.tracker.Receiver
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

import edu.rit.io.InStream;

/**
 * Class Receiver is the abstract base class for a thread that receives
 * messages. A receiver must be attached to a {@linkplain Proxy Proxy} before
 * the receiver can be used.
 *
 * @author  Alan Kaminsky
 * @version 04-Dec-2013
 */
public abstract class Receiver
	extends Thread
	{

// Hidden data members.

	byte opcode;
	Proxy proxy;
	Sender sender;
	InStream in;
	ReceiverListener listener;

// Exported constructors.

	/**
	 * Construct a new receiver. Failures are reported to the given listener.
	 *
	 * @param  listener  Receiver listener.
	 */
	public Receiver
		(ReceiverListener listener)
		{
		this.listener = listener;
		setContextClassLoader (Thread.currentThread().getContextClassLoader());
		//setDaemon (true);
		}

// Exported operations.

	/**
	 * Returns the remote host to which this receiver is connected.
	 *
	 * @return  Host name.
	 */
	public String host()
		{
		return proxy.host();
		}

	/**
	 * Returns the remote port to which this receiver is connected.
	 *
	 * @return  Port number.
	 */
	public int port()
		{
		return proxy.port();
		}

	/**
	 * Gracefully shut down communication with the far end. The
	 * <TT>shutdown()</TT> method blocks until the far end acknowledges that
	 * communication was shut down.
	 */
	public void shutdown()
		{
		proxy.shutdown();
		}

	/**
	 * Forcibly terminate communication with the far end. The
	 * <TT>terminate()</TT> method returns immediately, without waiting for an
	 * acknowledgment from the far end.
	 */
	public void terminate()
		{
		proxy.terminate();
		}

	}
