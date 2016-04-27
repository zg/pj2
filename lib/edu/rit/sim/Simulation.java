//******************************************************************************
//
// File:    Simulation.java
// Package: edu.rit.sim
// Unit:    Class edu.rit.sim.Simulation
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

package edu.rit.sim;

import edu.rit.util.PriorityQueue;

/**
 * Class Simulation provides a discrete event simulation. To write a discrete
 * event simulation program:
 * <OL TYPE=1>
 * <P><LI>
 * Create a Simulation object.
 * <P><LI>
 * Create one or more {@linkplain Event}s and add them to the simulation (by
 * calling the {@link #doAt(double,Event) doAt()} or {@link
 * #doAfter(double,Event) doAfter()} methods).
 * <P><LI>
 * Run the simulation (by calling the {@link #run() run()} method). The
 * simulation performs events, by calling each event's {@link Event#perform()
 * perform()} method, in order according to the events' simulation times, as
 * returned by each event's {@link Event#time() time()} method. Performing an
 * event may cause further events to be created and added to the simulation.
 * <P><LI>
 * When there are no more events, the simulation is finished. At this point the
 * simulation's {@link #run() run()} method returns.
 * </OL>
 *
 * @author  Alan Kaminsky
 * @version 12-Apr-2014
 */
public class Simulation
	{

// Hidden data members.

	// Minimum-priority queue of events.
	private PriorityQueue<Event> queue = new PriorityQueue<Event>();

	// Simulation time.
	private double T = 0.0;

// Exported constructors.

	/**
	 * Construct a new simulation.
	 */
	public Simulation()
		{
		}

// Exported operations.

	/**
	 * Returns the current simulation time.
	 *
	 * @return  Simulation time.
	 */
	public double time()
		{
		return T;
		}

	/**
	 * Schedule the given event to be performed at the given time in this
	 * simulation.
	 *
	 * @param  t      Simulation time for <TT>event</TT>.
	 * @param  event  Event to be performed.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>t</TT> is less than the current
	 *     simulation time.
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>event</TT> is null.
	 */
	public void doAt
		(double t,
		 Event event)
		{
		if (t < T)
			{
			throw new IllegalArgumentException
				("Simulation.doAt(): t = "+t+" less than simulation time ="+T+
				 ", illegal");
			}
		if (event == null)
			{
			throw new NullPointerException
				("Simulation.doAt(): event = null");
			}

		event.sim = this;
		event.time = t;
		queue.add (event);
		}

	/**
	 * Schedule the given event to be performed at a time <TT>dt</TT> in the
	 * future (at current simulation time + <TT>dt</TT>) in this simulation.
	 *
	 * @param  dt     Simulation time delta for <TT>event</TT>.
	 * @param  event  Event to be performed.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>dt</TT> is less than zero.
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>event</TT> is null.
	 */
	public void doAfter
		(double dt,
		 Event event)
		{
		doAt (T + dt, event);
		}

	/**
	 * Run the simulation. At the start of the simulation, the simulation time
	 * is 0. The <TT>run()</TT> method returns when there are no more events.
	 */
	public void run()
		{
		while (! queue.isEmpty())
			{
			Event event = queue.remove();
			T = event.time;
			event.perform();
			}
		}

	}
