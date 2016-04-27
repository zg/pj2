//******************************************************************************
//
// File:    PjmrJob.java
// Package: edu.rit.pjmr
// Unit:    Class edu.rit.pjmr.PjmrJob
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

package edu.rit.pjmr;

import edu.rit.pj2.Job;
import edu.rit.pj2.TaskSpec;
import edu.rit.pj2.Vbl;
import edu.rit.util.AList;

/**
 * Class PjmrJob is the abstract base class for a map-reduce job in the Parallel
 * Java Map Reduce Framework. A PJMR map-reduce job is designed to run on a
 * cluster of multicore nodes. A PJMR map-reduce job consists of one or more
 * {@linkplain MapperTask}s plus one {@linkplain ReducerTask}.
 * <P>
 * To program a PJMR map-reduce job:
 * <OL TYPE=1>
 * <P><LI>
 * Write {@linkplain Mapper}, {@linkplain Reducer}, {@linkplain Combiner}, and
 * {@linkplain Customizer} subclasses as appropriate for the map-reduce job.
 * <P><LI>
 * Write a subclass of class PjmrJob.
 * <P><LI>
 * In the PjmrJob subclass's {@link #main(String[]) main()} method:
 * <OL TYPE=a>
 * <P><LI>
 * Configure one or more mapper tasks. For each mapper task, call the {@link
 * #mapperTask() mapperTask()}, {@link #mapperTask(int) mapperTask(copies)}, or
 * {@link #mapperTask(String) mapperTask(nodeName)} method. The method
 * returns a {@linkplain MapperTaskConfig} object. Call further methods on the
 * {@linkplain MapperTaskConfig} object to configure the mapper task with one or
 * more {@linkplain Source Source}s, one or more {@linkplain Mapper}s, and an
 * optional {@linkplain Customizer}.
 * <P><LI>
 * Configure one reducer task. Call the {@link #reducerTask() reducerTask()}
 * method. The method returns a {@linkplain ReducerTaskConfig} object. Call
 * further methods on the {@linkplain ReducerTaskConfig} object to configure the
 * reducer task with one or more {@linkplain Reducer}s and/or a {@linkplain
 * Customizer}.
 * <P><LI>
 * Optionally, specify the combiner class that the mapper tasks and reducer task
 * will use. Call the {@link #combiner(Class) combiner()} method. If not
 * specified, the default is to use the {@linkplain Combiner} base class.
 * <P><LI>
 * As the last statement in the {@link #main(String[]) main()} method, call the
 * {@link #startJob() startJob()} method.
 * </OL>
 * </OL>
 * <P>
 * To run a PJMR map-reduce job, run the Parallel Java 2 launcher program,
 * {@link pj2 pj2}, specifying the PjmrJob subclass and any command line
 * arguments for the PjmrJob subclass's {@link #main(String[]) main()} method.
 * For example:
 * <PRE>
 *     $ java pj2 MyMapReduceJob <I>arg1</I> <I>arg2</I></PRE>
 * <P>
 * Each mapper task runs on a separate node of the cluster. Each source in each
 * mapper task generates data records to be analyzed; for example, the data
 * records might come from a file or files stored on the node's local hard disk.
 * Each mapper in each mapper task runs in a separate core on the node, in a
 * separate thread. A mapper may optionally be specified to use a GPU
 * coprocessor.
 * <P>
 * The reducer task runs on a separate node of the cluster. Optionally, the
 * reducer task may run in the PJMR job's process on the cluster frontend node.
 * Each reducer in the reducer task runs in a separate core on the node, in a
 * separate thread. A reducer may optionally be specified to use a GPU
 * coprocessor.
 * <P>
 * Once the PjmrJob's {@link #startJob() startJob()} method is called, the
 * mapper tasks and the reducer tasks all run in parallel. When all the mapper
 * and reducer tasks have finished, the PJMR job terminates.
 * <P>
 * See package {@linkplain edu.rit.pjmr.example edu.rit.pjmr.example} for
 * examples of PJMR map-reduce jobs.
 *
 * @param  <IK>  Mapper input key (data record ID) data type.
 * @param  <IV>  Mapper input value (data record content) data type.
 * @param  <OK>  Mapper output key (reducer input key) data type.
 * @param  <OV>  Mapper output value (reducer input value) data type; must
 *               implement interface {@linkplain edu.rit.pj2.Vbl Vbl}.
 *
 * @author  Alan Kaminsky
 * @version 13-Dec-2013
 */
public abstract class PjmrJob<IK,IV,OK,OV extends Vbl>
	extends Job
	{

// Hidden data members.

	private AList<MapperTaskConfig<IK,IV,OK,OV>> mapperTaskConfigList =
		new AList<MapperTaskConfig<IK,IV,OK,OV>>();

	private ReducerTaskConfig<OK,OV> reducerTaskConfig = null;

	private CombinerConfig<OK,OV> combinerConfig = null;

// Exported constructors.

	/**
	 * Construct a new PJMR job.
	 */
	public PjmrJob()
		{
		}

// Exported operations.

	/**
	 * Add one mapper task to this PJMR job. The mapper task may run on any
	 * node. To configure the mapper task, call methods on the returned
	 * {@linkplain MapperTaskConfig} object.
	 *
	 * @return  Mapper task config object.
	 */
	public MapperTaskConfig<IK,IV,OK,OV> mapperTask()
		{
		return mapperTask (1);
		}

	/**
	 * Add the given number of mapper tasks to this PJMR job. The mapper tasks
	 * may run on any nodes. To configure the mapper tasks, call methods on the
	 * returned {@linkplain MapperTaskConfig} object; all the mapper tasks will
	 * be configured the same.
	 *
	 * @param  copies  Number of mapper tasks (&ge; 1).
	 *
	 * @return  Mapper task config object.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>copies</TT> &lt; 1.
	 */
	public MapperTaskConfig<IK,IV,OK,OV> mapperTask
		(int copies)
		{
		if (copies < 1)
			throw new IllegalArgumentException (String.format
				("PjmrJob.mapperTask(): copies = %d illegal", copies));
		TaskSpec taskSpec = rule() .task (copies, MapperTask.class);
		MapperTaskConfig<IK,IV,OK,OV> mapperTaskConfig =
			new MapperTaskConfig<IK,IV,OK,OV> (taskSpec);
		for (int i = 0; i < copies; ++ i)
			mapperTaskConfigList.addLast (mapperTaskConfig);
		return mapperTaskConfig;
		}

	/**
	 * Add one mapper task to this PJMR job that will run on the given node. To
	 * configure the mapper task, call methods on the returned {@linkplain
	 * MapperTaskConfig} object.
	 *
	 * @param  nodeName  Node name.
	 *
	 * @return  Mapper task config object.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>nodeName</TT> is null.
	 */
	public MapperTaskConfig<IK,IV,OK,OV> mapperTask
		(String nodeName)
		{
		if (nodeName == null)
			throw new NullPointerException
				("PjmrJob.mapperTask(): nodeName is null");
		TaskSpec taskSpec = rule() .task (MapperTask.class);
		MapperTaskConfig<IK,IV,OK,OV> mapperTaskConfig =
			new MapperTaskConfig<IK,IV,OK,OV> (nodeName, taskSpec);
		mapperTaskConfigList.addLast (mapperTaskConfig);
		return mapperTaskConfig;
		}

	/**
	 * Specify the reducer task for this PJMR job. To configure the reducer
	 * task, call methods on the returned {@linkplain ReducerTaskConfig} object.
	 *
	 * @return  Reducer task config object.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if the reducer task has already been
	 *     specified.
	 */
	public ReducerTaskConfig<OK,OV> reducerTask()
		{
		if (reducerTaskConfig != null)
			throw new IllegalArgumentException
				("PjmrJob.reducerTask(): Reducer task already specified");
		TaskSpec taskSpec = rule() .task (ReducerTask.class);
		reducerTaskConfig = new ReducerTaskConfig<OK,OV> (taskSpec);
		return reducerTaskConfig;
		}

	/**
	 * Specify the combiner class for this PJMR job. If not specified, the
	 * default is the {@linkplain Combiner} base class.
	 *
	 * @param  combinerClass  Combiner class.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>combinerClass</TT> is null.
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if the combiner class has already been
	 *     specified.
	 */
	public <T extends Combiner<OK,OV>> void combiner
		(Class<T> combinerClass)
		{
		if (combinerConfig != null)
			throw new IllegalArgumentException
				("PjmrJob.combiner(): Combiner class already specified");
		combinerConfig = new CombinerConfig<OK,OV> (combinerClass);
		}

	/**
	 * Start this PJMR job.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if this PJMR job is misconfigured.
	 */
	public void startJob()
		{
		// Set default combiner class if necessary.
		if (combinerConfig == null)
			combinerConfig = new CombinerConfig<OK,OV> (Combiner.class);

		// Validate mapper tasks.
		int mapperTaskCount = mapperTaskConfigList.size();
		if (mapperTaskCount == 0)
			throw new IllegalArgumentException
				("PjmrJob.startJob(): No mapper tasks specified");
		for (int i = 0; i < mapperTaskCount; ++ i)
			mapperTaskConfigList.get(i).validate (this, combinerConfig);

		// Validate reducer task.
		if (reducerTaskConfig == null)
			throw new IllegalArgumentException
				("PjmrJob.startJob(): No reducer task specified");
		reducerTaskConfig.validate (this, combinerConfig, mapperTaskCount);
		}

	}
