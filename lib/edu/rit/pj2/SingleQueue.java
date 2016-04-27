//******************************************************************************
//
// File:    SingleQueue.java
// Package: edu.rit.pj2
// Unit:    Class edu.rit.pj2.SingleQueue
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

import java.util.concurrent.locks.LockSupport;

/**
 * Class SingleQueue provides a single-item queue. The queue has a maximum
 * capacity of one item. The queue's consumer must always be a certain thread,
 * specified to the constructor.
 *
 * @author  Alan Kaminsky
 * @version 19-May-2013
 */
class SingleQueue
	{

// Hidden data members.

	private Thread consumer;
	private volatile Object item;

// Exported constructors.

	/**
	 * Construct a new single-item queue.
	 *
	 * @param  consumer  Consumer thread.
	 */
	public SingleQueue
		(Thread consumer)
		{
		this.consumer = consumer;
		}

// Exported operations.

	/**
	 * Put the given item in this queue. Assumes the queue is empty. Assumes the
	 * item is not null. Non-blocking operation.
	 *
	 * @param  item  Item; assumed to be non-null.
	 */
	public void put
		(Object item)
		{
		this.item = item;
		LockSupport.unpark (consumer);
		}

	/**
	 * Get an item from this queue. Assumes the calling thread is the consumer
	 * thread specified to the constructor. Blocks until an item has been put
	 * into this queue. Assumes the calling thread will not be interrupted while
	 * blocked in this method.
	 *
	 * @return  Item.
	 */
	public Object get()
		{
		while (item == null)
			LockSupport.park (this);
		Object rv = item;
		item = null;
		return rv;
		}

// Unit test main program 1.

//	private static String strA;
//	private static String strB;
//	private static int delay;
//	private static int N;
//	private static Thread thrA;
//	private static Thread thrB;
//	private static SingleQueue AtoB;
//	private static SingleQueue BtoA;
//
//	public static void main
//		(String[] args)
//		{
//		if (args.length != 4) usage();
//		strA = args[0];
//		strB = args[1];
//		delay = Integer.parseInt (args[2]);
//		N = Integer.parseInt (args[3]);
//		thrA = new Thread()
//			{
//			public void run()
//				{
//				for (int i = 1; i <= N; ++ i)
//					{
//					try
//						{
//						Thread.sleep (delay);
//						}
//					catch (InterruptedException e) {}
//					System.out.printf ("Thread A sends \"%s\"%n", strA+" "+i);
//					AtoB.put (strA+" "+i);
//					Object rv = BtoA.get();
//					System.out.printf ("Thread A receives \"%s\"%n", rv);
//					}
//				}
//			};
//		Thread thrB = new Thread()
//			{
//			public void run()
//				{
//				for (int i = 1; i <= N; ++ i)
//					{
//					Object rv = AtoB.get();
//					System.out.printf ("Thread B receives \"%s\"%n", rv);
//					try
//						{
//						Thread.sleep (delay);
//						}
//					catch (InterruptedException e) {}
//					System.out.printf ("Thread B sends \"%s\"%n", strB+" "+i);
//					BtoA.put (strB+" "+i);
//					}
//				}
//			};
//		AtoB = new SingleQueue (thrB);
//		BtoA = new SingleQueue (thrA);
//		thrA.start();
//		thrB.start();
//		}
//
//	private static void usage()
//		{
//		System.err.println ("Usage: java edu.rit.pj2.SingleQueue <strA> <strB> <delay> <N>");
//		System.exit (1);
//		}

// Unit test main program 2.

//	private static String strA;
//	private static String strB;
//	private static int N;
//	private static Thread thrA;
//	private static Thread thrB;
//	private static SingleQueue AtoB;
//	private static SingleQueue BtoA;
//
//	public static void main
//		(String[] args)
//		{
//		if (args.length != 3) usage();
//		strA = args[0];
//		strB = args[1];
//		N = Integer.parseInt (args[2]);
//		thrA = new Thread()
//			{
//			public void run()
//				{
//				for (int i = 1; i <= N; ++ i)
//					{
//					System.out.printf ("Thread A sends \"%s\"%n", strA+" "+i);
//					AtoB.put (strA+" "+i);
//					Object rv = BtoA.get();
//					System.out.printf ("Thread A receives \"%s\"%n", rv);
//					}
//				}
//			};
//		Thread thrB = new Thread()
//			{
//			public void run()
//				{
//				for (int i = 1; i <= N; ++ i)
//					{
//					Object rv = AtoB.get();
//					System.out.printf ("Thread B receives \"%s\"%n", rv);
//					System.out.printf ("Thread B sends \"%s\"%n", strB+" "+i);
//					BtoA.put (strB+" "+i);
//					}
//				}
//			};
//		AtoB = new SingleQueue (thrB);
//		BtoA = new SingleQueue (thrA);
//		thrA.start();
//		thrB.start();
//		}
//
//	private static void usage()
//		{
//		System.err.println ("Usage: java edu.rit.pj2.SingleQueue <strA> <strB> <N>");
//		System.exit (1);
//		}

	}
