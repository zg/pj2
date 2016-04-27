//******************************************************************************
//
// File:    Team.java
// Package: edu.rit.pj2
// Unit:    Class edu.rit.pj2.Team
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

package edu.rit.pj2;

import edu.rit.util.AList;
import edu.rit.util.Predicate;

/**
 * Class Team provides a team of threads that are executing a {@linkplain
 * ParallelStatement}.
 *
 * @author  Alan Kaminsky
 * @version 18-Jul-2013
 */
class Team
	{

// Hidden class-wide data members.

	// List of already-created teams.
	private static AList<Team> pool = new AList<Team>();

// Hidden data members.

	// Number of threads in the team.
	private int NT;

	// The team threads themselves.
	private TeamThread[] thread;

	// Synchronization tree.
	// Each thread has a parent thread and zero or more child threads. Queues
	// (class SingleQueue) are used to synchronize and communicate between
	// threads.
	// fromParent[i] is the queue for thread i to receive from its parent.
	private SingleQueue[] fromParent;
	// toParent[i] is the queue for thread i to send to its parent.
	private SingleQueue[] toParent;
	// fromChild[i][j] is the queue for thread i to receive from its j-th child.
	private SingleQueue[][] fromChild;
	// toChild[i][j] is the queue for thread i to send to its j-th child.
	private SingleQueue[][] toChild;

	// Parallel statement the team is executing.
	private ParallelStatement statement;

// Hidden constructors.

	/**
	 * Construct a new team.
	 *
	 * @param  NT  Number of team threads. Assumed to be &ge; 1.
	 */
	private Team
		(int NT)
		{
//System.out.printf ("Creating new Team (%d)%n", NT);
//System.out.flush();
		// Create team threads.
		this.NT = NT;
		thread = new TeamThread [NT];
		for (int rank = 0; rank < NT; ++ rank)
			thread[rank] = new TeamThread (rank);

		// Set up synchronization tree.
		fromParent = new SingleQueue [NT];
		toParent = new SingleQueue [NT];
		fromChild = new SingleQueue [NT] [NT];
		toChild = new SingleQueue [NT] [NT];
		int[] numChildren = new int [NT];
		SingleQueue[] queues = null;
		fromParent[0] = new SingleQueue (thread[0]); // Special case
		for (int gap = 1; gap < NT; gap <<= 1)
			for (int parent = 0; parent < gap; ++ parent)
				{
				int child = parent + gap;
				if (child < NT)
					{
//System.out.printf ("Team(): parent = %d, child = %d%n", parent, child);
//System.out.flush();
					fromParent[child] =
						toChild[parent][numChildren[parent]] =
							new SingleQueue (thread[child]);
					toParent[child] =
						fromChild[parent][numChildren[parent]] =
							new SingleQueue (thread[parent]);
					++ numChildren[parent];
					}
				}
		for (int rank = 0; rank < NT; ++ rank)
			{
			queues = new SingleQueue [numChildren[rank]];
			System.arraycopy (fromChild[rank], 0, queues, 0, numChildren[rank]);
			fromChild[rank] = queues;
			queues = new SingleQueue [numChildren[rank]];
			System.arraycopy (toChild[rank], 0, queues, 0, numChildren[rank]);
			toChild[rank] = queues;
			}

		// Start team threads.
		for (int rank = 0; rank < NT; ++ rank)
			thread[rank].start();
		}

// Hidden operations.

	/**
	 * Get a team with the given number of threads from the pool. Construct a
	 * new team if necessary.
	 *
	 * @param  NT  Number of team threads.
	 */
	private static synchronized Team getTeam
		(final int NT)
		{
		int p = pool.position (new Predicate<Team>()
			{
			public boolean test (Team team)
				{
				return team.NT == NT;
				}
			});
		return p == -1 ? new Team (NT) : pool.swapRemoveLast (p);
		}

	/**
	 * Release the given team back to the pool.
	 *
	 * @param  team  Team.
	 */
	private static synchronized void releaseTeam
		(Team team)
		{
		pool.addLast (team);
		}

	/**
	 * Get a team with the given number of threads, and make every thread in the
	 * team execute the given parallel statement.
	 *
	 * @param  NT         Number of team threads. Must be &ge; 1.
	 * @param  statement  Parallel statement object.
	 *
	 * @param  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>NT</TT> &lt; 1.
	 */
	static void execute
		(int NT,
		 ParallelStatement statement)
		{
		if (NT < 1)
			throw new IllegalArgumentException (String.format
				("Team.execute(): NT = %d illegal", NT));
		Team team = getTeam (NT);
		team.execute (statement);
		releaseTeam (team);
		}

	/**
	 * Make every thread in this team execute the given parallel statement.
	 *
	 * @param  statement  Parallel statement object.
	 */
	private void execute
		(ParallelStatement statement)
		{
		this.statement = statement;
		toParent[0] = new SingleQueue (Thread.currentThread());
		fromParent[0].put (statement);
		Object result = toParent[0].get();
		if (result instanceof Throwable)
			throw new TerminateException
				("Exception in parallel team thread", (Throwable)result);
		((ReductionMap)result).setGlobalVariables();
		this.statement = null;
		}

// Hidden helper classes.

	/**
	 * Class TeamThread provides one thread in a team of threads that are
	 * executing a {@linkplain ParallelStatement}.
	 *
	 * @author  Alan Kaminsky
	 * @version 18-Jul-2013
	 */
	class TeamThread
		extends Thread
		{
		int rank;
		ReductionMap reductionMap = new ReductionMap();

		public TeamThread
			(int rank)
			{
			this.rank = rank;
			setDaemon (true);
			}

		public void run()
			{
			Throwable exc = null;
			SingleQueue fromParentQueue = fromParent[rank];
			SingleQueue[] toChildQueues = toChild[rank];
			SingleQueue[] fromChildQueues = fromChild[rank];
			// Note: The above are the same each time around.
			for (;;)
				{
				ParallelStatement statement = (ParallelStatement)
					fromParentQueue.get();
				for (SingleQueue queue: toChildQueues)
					queue.put (statement);
				reductionMap.clear();
				try
					{
					statement.run (rank, reductionMap);
					}
				catch (Throwable exc2)
					{
					exc = exc2;
					}
				for (SingleQueue queue : fromChildQueues)
					{
					Object result = queue.get();
					if (result instanceof Throwable)
						exc = (Throwable) result;
					if (exc == null)
						reductionMap.reduce ((ReductionMap)result);
					}
				toParent[rank].put (exc != null ? exc : reductionMap);
				// Note: The above might be different each time around.
				}
			}
		}

// Unit test programs.

//	/**
//	 * Unit test main program 1.
//	 */
//	public static void main
//		(String[] args)
//		throws Exception
//		{
//		new TestTask() .main (args);
//		}
//
//	private static class TestTask
//		extends Task
//		{
//		public void main
//			(String[] args)
//			throws Exception
//			{
//			if (args.length != 2) usage();
//			int iters = Integer.parseInt (args[0]);
//			int NT = Integer.parseInt (args[1]);
//			for (int i = 1; i <= iters; ++ i)
//				{
//				System.out.printf ("******** Begin iteration %d ********%n", i);
//				Team.execute (NT, new TestStatement (this));
//				System.out.printf ("******** End iteration %d **********%n", i);
//				Thread.sleep (1000L);
//				}
//			}
//		private static void usage()
//			{
//			System.err.println ("Usage: java edu.rit.pj2.Team <iters> <NT>");
//			System.exit (1);
//			}
//		}
//
//	private static class TestStatement
//		extends ParallelStatement
//		{
//		public TestStatement
//			(Task task)
//			{
//			super (task);
//			}
//		void run
//			(int rank)
//			{
//			System.out.printf ("Hello, world from %s%n",
//				Thread.currentThread());
//			}
//		}

	}
