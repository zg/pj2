//******************************************************************************
//
// File:    ReducerTaskConfig.java
// Package: edu.rit.pjmr
// Unit:    Class edu.rit.pjmr.ReducerTaskConfig
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

package edu.rit.pjmr;

import edu.rit.io.Streamable;
import edu.rit.io.InStream;
import edu.rit.io.OutStream;
import edu.rit.pj2.Debug;
import edu.rit.pj2.Job;
import edu.rit.pj2.Rule;
import edu.rit.pj2.Schedule;
import edu.rit.pj2.TaskSpec;
import edu.rit.pj2.Vbl;
import edu.rit.util.AList;
import java.io.IOException;
import java.util.Set;

/**
 * Class ReducerTaskConfig contains specifications for configuring a reducer
 * task.
 *
 * @param  <K>  Mapper output key data type.
 * @param  <V>  Mapper output value data type; must implement interface
 *              {@linkplain edu.rit.pj2.Vbl Vbl}.
 *
 * @author  Alan Kaminsky
 * @version 14-Mar-2014
 */
public class ReducerTaskConfig<K,V extends Vbl>
	implements Streamable
	{

// Exported data members.

	// List of reducer config objects.
	AList<ReducerConfig<K,V>> reducerConfigList =
		new AList<ReducerConfig<K,V>>();

	// Customizer config object.
	CustomizerConfig<K,V> customizerConfig = null;

	// Task specification.
	TaskSpec taskSpec;

// Exported constructors.

	/**
	 * Construct a new uninitialized reducer task config object. This
	 * constructor is for use only by object deserialization.
	 */
	ReducerTaskConfig()
		{
		}

	/**
	 * Construct a new reducer task config object.
	 *
	 * @param  taskSpec  Task specification.
	 */
	ReducerTaskConfig
		(TaskSpec taskSpec)
		{
		this.taskSpec = taskSpec;
		}

// Exported operations.

	/**
	 * Add the given {@linkplain Reducer} to this reducer task. The argument
	 * strings for the reducer, if any, may also be specified. A reducer task
	 * may have zero or more reducers. If a reducer task has no reducers, the
	 * reducer task should have a {@linkplain Customizer}.
	 *
	 * @param  reducerClass  Reducer class.
	 * @param  args          Zero or more argument strings for the reducer.
	 *
	 * @return  Reducer config object for <TT>reducerClass</TT>.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>reducerClass</TT> is null.
	 */
	public <T extends Reducer<K,V>> ReducerConfig<K,V> reducer
		(Class<T> reducerClass,
		 String... args)
		{
		return reducer (1, reducerClass, args);
		}

	/**
	 * Add the given number of copies of the given {@linkplain Reducer} to this
	 * reducer task. The argument strings for the reducers, if any, may also be
	 * specified; each reducer has the same argument strings. A reducer task may
	 * have zero or more reducers. If a reducer task has no reducers, the
	 * reducer task should have a {@linkplain Customizer}.
	 *
	 * @param  copies        Number of copies to add (&ge; 1).
	 * @param  reducerClass  Reducer class.
	 * @param  args          Zero or more argument strings for the reducers.
	 *
	 * @return  Reducer config object for <TT>reducerClass</TT>.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>copies</TT> &lt; 1.
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>reducerClass</TT> is null.
	 */
	public <T extends Reducer<K,V>> ReducerConfig<K,V> reducer
		(int copies,
		 Class<T> reducerClass,
		 String... args)
		{
		if (copies < 1)
			throw new IllegalArgumentException (String.format
				("ReducerTaskConfig.reducer(): copies = %d illegal", copies));
		ReducerConfig<K,V> reducerConfig =
			new ReducerConfig<K,V> (this, reducerClass, args);
		for (int i = 0; i < copies; ++ i)
			reducerConfigList.addLast (reducerConfig);
		return reducerConfig;
		}

	/**
	 * Specify the given {@linkplain Customizer} for this reducer task. The
	 * argument strings for the customizer, if any, may also be specified. The
	 * default is not to use a customizer. If a reducer task has no customizer,
	 * the reducer task should have at least one {@linkplain Reducer}.
	 *
	 * @param  customizerClass  Customizer class.
	 * @param  args             Zero or more argument strings for the
	 *                          customizer.
	 *
	 * @return  This reducer task config object.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>customizerClass</TT> is null.
	 */
	public <T extends Customizer<K,V>> ReducerTaskConfig<K,V> customizer
		(Class<T> customizerClass,
		 String... args)
		{
		customizerConfig = new CustomizerConfig (customizerClass, args);
		return this;
		}

	/**
	 * Specify this reducer task's <TT>threads</TT> property. For further
	 * information, see {@link edu.rit.pj2.TaskSpec#threads(int)
	 * TaskSpec.threads()}.
	 *
	 * @param  threads  Number of threads (&ge; 1).
	 *
	 * @return  This reducer task config object.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>threads</TT> is illegal.
	 */
	public ReducerTaskConfig<K,V> threads
		(int threads)
		{
		taskSpec.threads (threads);
		return this;
		}

	/**
	 * Specify this reducer task's <TT>schedule</TT> property. For further
	 * information, see {@link edu.rit.pj2.TaskSpec#schedule(Schedule)
	 * TaskSpec.schedule()}.
	 *
	 * @param  schedule  Parallel for loop schedule.
	 *
	 * @return  This reducer task config object.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>schedule</TT> is null.
	 */
	public ReducerTaskConfig<K,V> schedule
		(Schedule schedule)
		{
		taskSpec.schedule (schedule);
		return this;
		}

	/**
	 * Specify this reducer task's <TT>chunk</TT> property. For further
	 * information, see {@link edu.rit.pj2.TaskSpec#chunk(int)
	 * TaskSpec.chunk()}.
	 *
	 * @param  chunk  Chunk size (&ge; 1).
	 *
	 * @return  This reducer task config object.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>chunk</TT> is illegal.
	 */
	public ReducerTaskConfig<K,V> chunk
		(int chunk)
		{
		taskSpec.chunk (chunk);
		return this;
		}

	/**
	 * Specify the JVM flags for this reducer task. For further information, see
	 * {@link edu.rit.pj2.TaskSpec#jvmFlags(String[]) TaskSpec.jvmFlags()}.
	 *
	 * @param  jvmFlags  JVM flags (zero or more).
	 *
	 * @return  This reducer task config object.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>jvmFlags</TT> is null or any
	 *     element of <TT>jvmFlags</TT> is null.
	 */
	public ReducerTaskConfig<K,V> jvmFlags
		(String... jvmFlags)
		{
		taskSpec.jvmFlags (jvmFlags);
		return this;
		}

	/**
	 * Specify that this reducer task must run in the job process. For further
	 * information, see {@link edu.rit.pj2.TaskSpec#runInJobProcess()
	 * TaskSpec.runInJobProcess()}.
	 *
	 * @return  This reducer task config object.
	 */
	public ReducerTaskConfig<K,V> runInJobProcess()
		{
		taskSpec.runInJobProcess();
		return this;
		}

	/**
	 * Print the given debugging messages for this reducer task. For further
	 * information, see {@link edu.rit.pj2.TaskSpec#debug(Debug[])
	 * TaskSpec.debug()}.
	 *
	 * @param  debug  Debugging message(s) to print, or null to print none.
	 *
	 * @return  This reducer task config object.
	 */
	public ReducerTaskConfig<K,V> debug
		(Debug... debug)
		{
		taskSpec.debug (debug);
		return this;
		}

	/**
	 * Print the given debugging messages for this reducer task. For further
	 * information, see {@link edu.rit.pj2.TaskSpec#debug(Debug[])
	 * TaskSpec.debug()}.
	 *
	 * @param  debug  Set of debugging message(s) to print, or null to print
	 *                none.
	 *
	 * @return  This reducer task config object.
	 */
	public ReducerTaskConfig<K,V> debug
		(Set<Debug> debug)
		{
		taskSpec.debug (debug);
		return this;
		}

	/**
	 * Write this reducer task config object to the given out stream.
	 *
	 * @param  out  Out stream.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void writeOut
		(OutStream out)
		throws IOException
		{
		out.writeReference (reducerConfigList);
		out.clearCache();
		out.writeFields (customizerConfig);
		}

	/**
	 * Read this combiner config object from the given in stream.
	 *
	 * @param  in  In stream.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void readIn
		(InStream in)
		throws IOException
		{
		reducerConfigList = (AList<ReducerConfig<K,V>>) in.readReference();
		in.clearCache();
		customizerConfig = in.readFields (new CustomizerConfig<K,V>());
		}

// Hidden operations.

	/**
	 * Validate this reducer task config object. If valid, the appropriate tuple
	 * is put into tuple space. If invalid, an exception is thrown with a detail
	 * message describing the error.
	 *
	 * @param  job              Job.
	 * @param  combinerConfig   Combiner config object.
	 * @param  mapperTaskCount  Number of mapper tasks.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if this reducer task config object is
	 *     invalid.
	 */
	void validate
		(Job job,
		 CombinerConfig<K,V> combinerConfig,
		 int mapperTaskCount)
		{
		// Validate number of reducers and/or customizer.
		int reducerCount = reducerConfigList.size();
		if (reducerCount == 0 && customizerConfig == null)
			throw new IllegalArgumentException
				("ReducerTaskConfig.validate(): At least one reducer or customizer must be specified");

		// Count reducers and GPUs needed.
		int gpuCount = 0;
		for (int i = 0; i < reducerCount; ++ i)
			{
			ReducerConfig<K,V> reducerConfig = reducerConfigList.get (i);
			if (reducerConfig.needsGpu)
				++ gpuCount;
			}

		// If no customizer was specified, use a default customizer.
		if (customizerConfig == null)
			customizerConfig = new CustomizerConfig<K,V> (Customizer.class);

		// Set up task specification.
		taskSpec
			.args (""+mapperTaskCount)
			.cores (Math.max (reducerCount, 1))
			.gpus (gpuCount);

		// Put reducer task config tuple into tuple space.
		job.putTuple (new ReducerTaskConfigTuple (this, combinerConfig));
		}

	}
