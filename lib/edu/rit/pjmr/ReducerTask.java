//******************************************************************************
//
// File:    ReducerTask.java
// Package: edu.rit.pjmr
// Unit:    Class edu.rit.pjmr.ReducerTask
//
// This Java source file is copyright (C) 2015 by Alan Kaminsky. All rights
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

package edu.rit.pjmr;

import edu.rit.pj2.Loop;
import edu.rit.pj2.Task;
import edu.rit.pj2.Vbl;
import edu.rit.util.Action;
import edu.rit.util.Pair;
import edu.rit.util.Sorting;

/**
 * Class ReducerTask provides a reducer task in the Parallel Java Map Reduce
 * Framework. Do not create reducer tasks directly; rather, define reducer tasks
 * as part of a {@linkplain PjmrJob}.
 * <P>
 * Each reducer task in a PJMR job does the following, using the configuration
 * specified in the PJMR job. For further information, refer to the
 * documentation for the classes mentioned below.
 * <OL TYPE=1>
 * <P><LI>
 * For each of the reducer task's configured {@linkplain Reducer}s, create the
 * reducer object.
 * <P><LI>
 * Create the reducer task's configured {@linkplain Combiner}.
 * <P><LI>
 * If the reducer task was configured with a {@linkplain Customizer}, create the
 * customizer object and call its <TT>start()</TT> method, passing in any
 * configured argument strings plus the combiner.
 * <P><LI>
 * Receive a combiner object from each {@linkplain MapperTask}. Reduce all the
 * received combiner objects together into the reducer task's combiner.
 * <P><LI>
 * If the reducer task was configured with a customizer and the customizer's
 * <TT>comesBefore()</TT> method was overridden, sort the combiner's (key,
 * value) pairs into the order determined by the <TT>comesBefore()</TT> method.
 * Otherwise, leave the combiner's (key, value) pairs in an unspecified order.
 * <P><LI>
 * Do a parallel loop over the (key, value) pairs in the combiner. Each parallel
 * loop iteration processes one (key, value) pair. There is one parallel team
 * thread for each reducer. The reducer task's <TT>schedule</TT> and
 * <TT>chunk</TT> properties determine how the parallel loop iterations are
 * partitioned among the parallel team threads; that is, how the (key, value)
 * pairs are partitioned among the reducers. Each parallel team thread does the
 * following:
 * <OL TYPE=a>
 * <P><LI>
 * Call the reducer's <TT>start()</TT> method, passing in any configured
 * argument strings.
 * <P><LI>
 * Repeatedly call the reducer's <TT>reduce()</TT> method, passing in one of the
 * combiner's (key, value) pairs.
 * <P><LI>
 * When there are no more pairs, call the reducer's <TT>finish()</TT> method.
 * </OL>
 * <P><LI>
 * After all the parallel team threads have finished, if the reducer task was
 * configured with a customizer, call the customizer's <TT>finish()</TT> method,
 * passing in the combiner.
 * </OL>
 *
 * @param  <K>  Mapper output key data type.
 * @param  <V>  Mapper output value data type; must implement interface
 *              {@linkplain edu.rit.pj2.Vbl Vbl}.
 *
 * @author  Alan Kaminsky
 * @version 12-Jan-2015
 */
public class ReducerTask<K,V extends Vbl>
	extends Task
	{

// Global variables.

	// Combiner, its size, array of key-value pairs.
	Combiner<K,V> combiner;
	int n;
	Pair<K,V>[] pairs;

	// Reducer and reducer argument strings for each parallel team thread.
	Reducer<K,V>[] thrReducer;
	String[][] thrReducerArgs;

	// Customizer.
	Customizer<K,V> customizer;

// Task main program.

	/**
	 * Reducer task main program.
	 *
	 * @param  args  Array of command line argument strings.
	 * <UL>
	 * <LI><TT>args[0]</TT> = Number of mapper tasks
	 * </UL>
	 */
	public void main
		(String[] args)
		throws Exception
		{
		// Get number of mapper tasks.
		int mapperTaskCount = Integer.parseInt (args[0]);

		// Get configuration.
		ReducerTaskConfigTuple<K,V> configTuple =
			(ReducerTaskConfigTuple<K,V>) takeTuple
				(new ReducerTaskConfigTuple<K,V>());
		ReducerTaskConfig<K,V> config = configTuple.config;

		// Get number of reducers.
		int reducerCount = config.reducerConfigList.size();

		// Set up reducers for each parallel team thread.
		thrReducer = (Reducer<K,V>[]) new Reducer [reducerCount];
		thrReducerArgs = new String [reducerCount] [];
		int thr = 0;
		int m = config.reducerConfigList.size();
		for (int i = 0; i < m; ++ i)
			{
			ReducerConfig<K,V> reducerConfig = config.reducerConfigList.get (i);
			thrReducer[thr] = reducerConfig.newInstance();
			thrReducerArgs[thr] = reducerConfig.reducerArgs;
			++ thr;
			}

		// Set up combiner.
		combiner = configTuple.combinerConfig.newInstance();

		// Set up customizer.
		customizer = config.customizerConfig.newInstance();
		customizer.start (config.customizerConfig.customizerArgs, combiner);

		// Determine whether to sort the combiner's pairs.
		boolean doSort;
		try
			{
			customizer.comesBefore (null, null, null, null);
			doSort = true;
			}
		catch (UnsupportedOperationException exc)
			{
			doSort = false;
			}
		catch (Throwable exc)
			{
			doSort = true;
			}

		// Receive and accumulate pairs from mapper tasks.
		PairReceiver<K,V> pairReceiver =
			new PairReceiver<K,V> (this, mapperTaskCount);
		pairReceiver.forEachPairDo (new Action<Pair<K,V>>()
			{
			public void run (Pair<K,V> pair)
				{
				combiner.add (pair.key(), pair.value());
				}
			});

		// Get arrays of keys and values; sort them if necessary.
		n = combiner.size();
		pairs = combiner.toArray ((Pair<K,V>[]) new Pair [n]);
		if (doSort)
			Sorting.sort (pairs, new Sorting.Object<Pair<K,V>>()
				{
				public boolean comesBefore (Pair<K,V>[] x, int a, int b)
					{
					return customizer.comesBefore
						(x[a].key(), x[a].value(), x[b].key(), x[b].value());
					}
				});

		// Run reducers in parallel.
		if (reducerCount > 0)
			parallelFor (0, n - 1) .threads (reducerCount) .exec (new Loop()
				{
				Reducer<K,V> reducer;
				public void start()
					{
					reducer = thrReducer[rank()];
					reducer.start (thrReducerArgs[rank()]);
					}
				public void run (int i)
					{
					reducer.reduce (pairs[i].key(), pairs[i].value());
					}
				public void finish()
					{
					reducer.finish();
					}
				});

		// Finish customizer.
		customizer.finish (combiner);
		}

	}
