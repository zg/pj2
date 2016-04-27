//******************************************************************************
//
// File:    MapperTaskConfig.java
// Package: edu.rit.pjmr
// Unit:    Class edu.rit.pjmr.MapperTaskConfig
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
 * Class MapperTaskConfig contains specifications for configuring a mapper task.
 *
 * @param  <IK>  Mapper input key data type.
 * @param  <IV>  Mapper input value data type.
 * @param  <OK>  Mapper output key data type.
 * @param  <OV>  Mapper output value data type; must implement interface
 *               {@linkplain edu.rit.pj2.Vbl Vbl}.
 *
 * @author  Alan Kaminsky
 * @version 14-Mar-2014
 */
public class MapperTaskConfig<IK,IV,OK,OV extends Vbl>
	implements Streamable
	{

// Exported data members.

	// Node name.
	String nodeName = TaskSpec.DEFAULT_NODE_NAME;

	// Total number of mappers.
	int mapperCount = 0;

	// List of source config objects.
	AList<SourceConfig<IK,IV,OK,OV>> sourceConfigList =
		new AList<SourceConfig<IK,IV,OK,OV>>();

	// Customizer config object.
	CustomizerConfig<OK,OV> customizerConfig = null;

	// Task specification.
	TaskSpec taskSpec;

// Exported constructors.

	/**
	 * Construct a new uninitialized mapper task config object. This constructor
	 * is for use only by object deserialization.
	 */
	MapperTaskConfig()
		{
		}

	/**
	 * Construct a new mapper task config object. The mapper task may run on any
	 * node.
	 *
	 * @param  taskSpec  Task specification.
	 */
	MapperTaskConfig
		(TaskSpec taskSpec)
		{
		this.taskSpec = taskSpec;
		}

	/**
	 * Construct a new mapper task config object. The mapper task must run on
	 * the given node.
	 *
	 * @param  nodeName  Node name.
	 * @param  taskSpec  Task specification.
	 */
	MapperTaskConfig
		(String nodeName,
		 TaskSpec taskSpec)
		{
		this.nodeName = nodeName;
		this.taskSpec = taskSpec;
		}

// Exported operations.

	/**
	 * Add the given {@linkplain Source} to this mapper task. A mapper task must
	 * include at least one source; it may include more than one source. After
	 * calling <TT>source()</TT>, call {@link
	 * SourceConfig#mapper(Class,String[]) mapper()} on the returned {@linkplain
	 * SourceConfig} object to attach one or more mappers to the
	 * <TT>source</TT>.
	 *
	 * @param  source  Source.
	 *
	 * @return  Source config object for <TT>source</TT>.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>source</TT> is null.
	 */
	public SourceConfig<IK,IV,OK,OV> source
		(Source<IK,IV> source)
		{
		if (source == null)
			throw new NullPointerException
				("MapperTaskConfig.source(): source is null");
		SourceConfig<IK,IV,OK,OV> sc =
			new SourceConfig<IK,IV,OK,OV> (this, source);
		sourceConfigList.addLast (sc);
		return sc;
		}

	/**
	 * Specify the given {@linkplain Customizer} for this mapper task. The
	 * argument strings for the customizer, if any, may also be specified. The
	 * default is not to use a customizer.
	 *
	 * @param  customizerClass  Customizer class.
	 * @param  args             Zero or more argument strings for the
	 *                          customizer.
	 *
	 * @return  This mapper task config object.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>customizerClass</TT> is null.
	 */
	public <T extends Customizer<OK,OV>>
	MapperTaskConfig<IK,IV,OK,OV> customizer
		(Class<T> customizerClass,
		 String... args)
		{
		customizerConfig = new CustomizerConfig (customizerClass, args);
		return this;
		}

	/**
	 * Specify this mapper task's <TT>threads</TT> property. For further
	 * information, see {@link edu.rit.pj2.TaskSpec#threads(int)
	 * TaskSpec.threads()}.
	 *
	 * @param  threads  Number of threads (&ge; 1).
	 *
	 * @return  This mapper task config object.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>threads</TT> is illegal.
	 */
	public MapperTaskConfig<IK,IV,OK,OV> threads
		(int threads)
		{
		taskSpec.threads (threads);
		return this;
		}

	/**
	 * Specify this mapper task's <TT>schedule</TT> property. For further
	 * information, see {@link edu.rit.pj2.TaskSpec#schedule(Schedule)
	 * TaskSpec.schedule()}.
	 *
	 * @param  schedule  Parallel for loop schedule.
	 *
	 * @return  This mapper task config object.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>schedule</TT> is null.
	 */
	public MapperTaskConfig<IK,IV,OK,OV> schedule
		(Schedule schedule)
		{
		taskSpec.schedule (schedule);
		return this;
		}

	/**
	 * Specify this mapper task's <TT>chunk</TT> property. For further
	 * information, see {@link edu.rit.pj2.TaskSpec#chunk(int)
	 * TaskSpec.chunk()}.
	 *
	 * @param  chunk  Chunk size (&ge; 1).
	 *
	 * @return  This mapper task config object.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>chunk</TT> is illegal.
	 */
	public MapperTaskConfig<IK,IV,OK,OV> chunk
		(int chunk)
		{
		taskSpec.chunk (chunk);
		return this;
		}

	/**
	 * Specify the JVM flags for this mapper task. For further information, see
	 * {@link edu.rit.pj2.TaskSpec#jvmFlags(String[]) TaskSpec.jvmFlags()}.
	 *
	 * @param  jvmFlags  JVM flags (zero or more).
	 *
	 * @return  This mapper task config object.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>jvmFlags</TT> is null or any
	 *     element of <TT>jvmFlags</TT> is null.
	 */
	public MapperTaskConfig<IK,IV,OK,OV> jvmFlags
		(String... jvmFlags)
		{
		taskSpec.jvmFlags (jvmFlags);
		return this;
		}

	/**
	 * Print the given debugging messages for this mapper task. For further
	 * information, see {@link edu.rit.pj2.TaskSpec#debug(Debug[])
	 * TaskSpec.debug()}.
	 *
	 * @param  debug  Debugging message(s) to print, or null to print none.
	 *
	 * @return  This mapper task config object.
	 */
	public MapperTaskConfig<IK,IV,OK,OV> debug
		(Debug... debug)
		{
		taskSpec.debug (debug);
		return this;
		}

	/**
	 * Print the given debugging messages for this mapper task. For further
	 * information, see {@link edu.rit.pj2.TaskSpec#debug(Debug[])
	 * TaskSpec.debug()}.
	 *
	 * @param  debug  Set of debugging message(s) to print, or null to print
	 *                none.
	 *
	 * @return  This mapper task config object.
	 */
	public MapperTaskConfig<IK,IV,OK,OV> debug
		(Set<Debug> debug)
		{
		taskSpec.debug (debug);
		return this;
		}

	/**
	 * Write this mapper task config object to the given out stream.
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
		out.writeString (nodeName);
		out.writeInt (mapperCount);
		out.writeReference (sourceConfigList);
		out.clearCache();
		out.writeFields (customizerConfig);
		}

	/**
	 * Read this mapper task config object from the given in stream.
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
		nodeName = in.readString();
		mapperCount = in.readInt();
		sourceConfigList = (AList<SourceConfig<IK,IV,OK,OV>>)
			in.readReference();
		in.clearCache();
		customizerConfig = in.readFields (new CustomizerConfig<OK,OV>());
		}

// Hidden operations.

	/**
	 * Validate this mapper task config object. If valid, the appropriate tuple
	 * is put into the given job's tuple space. If invalid, an exception is
	 * thrown with a detail message describing the error.
	 *
	 * @param  job             Job.
	 * @param  combinerConfig  Combiner config object.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if this mapper task config object is
	 *     invalid.
	 */
	void validate
		(Job job,
		 CombinerConfig<OK,OV> combinerConfig)
		{
		// Validate number of sources.
		int sourceCount = sourceConfigList.size();
		if (sourceCount == 0)
			throw new IllegalArgumentException
				("MapperTaskConfig.validate(): At least one source must be specified");

		// Validate number of mappers for each source. Count mappers and GPUs
		// needed.
		mapperCount = 0;
		int gpuCount = 0;
		for (int i = 0; i < sourceCount; ++ i)
			{
			SourceConfig<IK,IV,OK,OV> sourceConfig = sourceConfigList.get (i);
			int sourceMapperCount = sourceConfig.mapperConfigList.size();
			if (sourceMapperCount == 0)
				throw new IllegalArgumentException (String.format
					("MapperTaskConfig.validate(): At least one mapper must be specified for source %d", i + 1));
			mapperCount += sourceMapperCount;
			for (int j = 0; j < sourceMapperCount; ++ j)
				{
				MapperConfig<IK,IV,OK,OV> mapperConfig =
					sourceConfig.mapperConfigList.get (j);
				if (mapperConfig.needsGpu)
					++ gpuCount;
				}
			}

		// If no customizer was specified, use a default customizer.
		if (customizerConfig == null)
			customizerConfig = new CustomizerConfig<OK,OV> (Customizer.class);

		// If node name was defaulted, run task on any node.
		if (nodeName == TaskSpec.DEFAULT_NODE_NAME)
			nodeName = TaskSpec.ANY_NODE_NAME;

		// Set up task specification.
		taskSpec
			.args (nodeName)
			.nodeName (nodeName)
			.cores (mapperCount)
			.gpus (gpuCount);

		// Put mapper task config tuple into tuple space.
		job.putTuple (new MapperTaskConfigTuple (this, combinerConfig));
		}

	}
