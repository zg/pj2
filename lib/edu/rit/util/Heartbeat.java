//******************************************************************************
//
// File:    Heartbeat.java
// Package: edu.rit.util
// Unit:    Class edu.rit.util.Heartbeat
//
// This Java source file is copyright (C) 2014 by Alan Kaminsky. All rights
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

package edu.rit.util;

import java.io.IOException;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;

/**
 * Class Heartbeat is the abstract base class for an object that sends
 * heartbeats to and receives heartbeats from some entity.
 * <P>
 * When a heartbeat object is constructed, the <I>heartbeat interval</I> is
 * specified. The default heartbeat interval is 30 seconds. The heartbeat object
 * will send heartbeats to the other entity periodically at the heartbeat
 * interval. If the heartbeat object fails to receive a heartbeat from the other
 * entity within 1.5 times the heartbeat interval since receiving the last
 * heartbeat, the heartbeat object decides that the other entity died.
 * <P>
 * After constructing a heartbeat object, the heartbeats do not actually start
 * until the {@link #enable(ScheduledExecutorService) enable()} method is
 * called.
 *
 * @author  Alan Kaminsky
 * @version 21-Jul-2014
 */
public abstract class Heartbeat
	{

// Hidden data members.

	// Executor for doing timed actions.
	private ScheduledExecutorService executor;

	// Heartbeat and loss detection intervals.
	private long heartbeatInterval;
	private long lossInterval;
	private TimeUnit timeUnit;
	private boolean canceled;

	// For sending heartbeats periodically to the other entity.
	private ScheduledFuture<?> sender;
	private Runnable senderTask = new Runnable()
		{
		public void run()
			{
			doSendHeartbeat();
			}
		};

	// For detecting loss of heartbeats from the other entity.
	private ScheduledFuture<?> lossDetector;
	private Runnable lossDetectorTask = new Runnable()
		{
		public void run()
			{
			doDied();
			}
		};

// Exported constructors.

	/**
	 * Construct a new heartbeat object with the default heartbeat interval (30
	 * seconds).
	 */
	public Heartbeat()
		{
		this (30L, TimeUnit.SECONDS);
		}

	/**
	 * Construct a new heartbeat object with the given heartbeat interval.
	 *
	 * @param  interval  Heartbeat interval.
	 * @param  timeUnit  Heartbeat interval time unit.
	 */
	public Heartbeat
		(long interval,
		 TimeUnit timeUnit)
		{
		this.heartbeatInterval = interval;
		this.lossInterval = interval*3L/2L;
		this.timeUnit = timeUnit;
		}

// Exported operations.

	/**
	 * Enable this heartbeat object to start exchanging heartbeats.
	 *
	 * @param  executor  Executor for doing timed actions. If null, this
	 *                   heartbeat object is not enabled.
	 */
	public synchronized void enable
		(ScheduledExecutorService executor)
		{
		if (executor != null)
			{
			this.executor = executor;
			sender = executor.scheduleAtFixedRate
				(senderTask, (heartbeatInterval + 9L)/10L,
				 heartbeatInterval, timeUnit);
			lossDetector = executor.schedule
				(lossDetectorTask, lossInterval, timeUnit);
			}
		}

	/**
	 * Report that a heartbeat was received from the other entity.
	 */
	public synchronized void receiveHeartbeat()
		{
		if (lossDetector != null)
			lossDetector.cancel (false);
		lossDetector = null;
		if (executor != null)
			lossDetector = executor.schedule
				(lossDetectorTask, lossInterval, timeUnit);
		}

	/**
	 * Stop exchanging heartbeats with the other entity.
	 */
	public synchronized void cancel()
		{
		if (sender != null)
			sender.cancel (false);
		sender = null;
		if (lossDetector != null)
			lossDetector.cancel (false);
		lossDetector = null;
		canceled = true;
		}

// Hidden operations.

	/**
	 * Send a heartbeat to the other entity.
	 */
	private synchronized void doSendHeartbeat()
		{
		try
			{
			if (! canceled) sendHeartbeat();
			}
		catch (Throwable exc)
			{
			}
		}

	/**
	 * Take action when this heartbeat object detects loss of heartbeats from
	 * the other entity.
	 */
	private synchronized void doDied()
		{
		if (! canceled) died();
		}

	/**
	 * Send a heartbeat to the other entity. This method must be overridden in a
	 * subclass. Any exception thrown by this method is ignored.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	protected abstract void sendHeartbeat()
		throws IOException;

	/**
	 * Take action when this heartbeat object detects loss of heartbeats from
	 * the other entity. This method must be overridden in a subclass.
	 */
	protected abstract void died();

	}
