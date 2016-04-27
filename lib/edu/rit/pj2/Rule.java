//******************************************************************************
//
// File:    Rule.java
// Package: edu.rit.pj2
// Unit:    Class edu.rit.pj2.Rule
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

package edu.rit.pj2;

import edu.rit.pj2.tracker.JobProperties;
import edu.rit.util.AList;
import edu.rit.util.DList;
import edu.rit.util.DListEntry;
import java.util.Iterator;

/**
 * Class Rule provides a PJ2 rule. A PJ2 parallel program is either a single
 * task (a subclass of class {@linkplain Task Task}) or is a number of tasks
 * grouped together into a job (a subclass of class {@linkplain Job Job}). In
 * the latter case, the job consists of a number of rules, which are instances
 * of class Rule or a subclass thereof.
 * <P>
 * A PJ2 job maintains a repository of {@linkplain Tuple Tuple}s known as
 * <I>tuple space.</I> The job's execution is driven by <I>firing</I> the rules,
 * based on tuples that are written into tuple space.
 * <P>
 * <B>Programming rules.</B>
 * To create a rule:
 * <UL>
 * <P><LI>
 * Obtain an instance of class Rule by calling the {@link Job#rule() rule()}
 * method in the {@link Job#main(String[]) main()} method of a {@linkplain Job}.
 * <P><LI>
 * Specify the condition for firing the rule, one of the following:
 * <UL>
 * <P><LI>
 * <I>Start rule</I>&mdash;To fire the rule once when the job starts, call the
 * {@link #atStart() atStart()} method. If the firing condition is not
 * specified, the rule is a start rule by default.
 * <P><LI>
 * <I>On-demand rule</I>&mdash;To fire the rule when a certain tuple or tuples
 * are present in tuple space, call the {@link #whenMatch(Tuple) whenMatch()}
 * method one or more times, specifying the template or templates to match
 * against the tuples.
 * <P>
 * Optionally, call the {@link #matcher(Matcher) matcher()} method to override
 * the default tuple matching algorithm. See below for further information.
 * <P><LI>
 * <I>Finish rule</I>&mdash;To fire the rule once when the job finishes, call
 * the {@link #atFinish() atFinish()} method.
 * </UL>
 * <P><LI>
 * Call the {@link #task(Class) task(Class)} method and/or the {@link
 * #task(int,Class) task(int,Class)} method one or more times to specify the
 * task or tasks in the rule's <I>task group.</I> When the rule fires, all of
 * the rule's tasks will be executed as a single group. For further information
 * about creating a task specification, see class {@linkplain TaskSpec
 * TaskSpec}.
 * </UL>
 * <P>
 * <B>Rule execution.</B>
 * The rules are fired as follows:
 * <UL>
 * <P><LI>
 * For a start rule, the rule is fired once when the job starts. For each task
 * in the rule's task group, an instance of the task subclass is constructed,
 * and the task's {@link Task#main(String[]) main()} method is called, passing
 * in the task's command line arguments. The task may read and take tuples from
 * tuple space and put tuples into tuple space.
 * <P><LI>
 * For an on-demand rule, the rule is fired whenever tuples exist in tuple space
 * that <I>match</I> the rule's templates. An on-demand rule may fire more than
 * once during the course of a job. When the rule fires, the matching tuples are
 * automatically taken out of tuple space. For each task in the rule's task
 * group, an instance of the task subclass is constructed, and the task's {@link
 * Task#main(String[]) main()} method is called, passing in the task's command
 * line arguments. The task may obtain the matching tuples that caused the
 * on-demand rule to fire. The task may also read and take tuples from tuple
 * space and put tuples into tuple space.
 * <P>
 * The default algorithm to decide whether there are tuples in tuple space that
 * match the templates in the rule is to consider each template separately, in
 * the order the templates were added to the rule. For each template, the tuples
 * in tuple space are considered in the order the tuples were added to tuple
 * space. If the template matches a tuple, as determined by the template's
 * {@link Tuple#match(Tuple) match()} method, the matching tuple is remembered,
 * and the next template is considered. Each template must match a
 * <I>different</I> tuple. If a matching tuple cannot be found for one or more
 * templates, the rule is not fired. If a matching tuple is found for every
 * template, the matching tuples are removed from tuple space, the rule is
 * fired, and copies of the matching tuples are provided to each of the tasks in
 * the task group.
 * <P>
 * The default matching algorithm can be changed for a given rule by calling the
 * {@link #matcher(Matcher) matcher()} method.
 * <P><LI>
 * For a finish rule, the rule is fired once when the job finishes (when all
 * other tasks have finished). For each task in the rule's task group, an
 * instance of the task subclass is constructed, and the task's {@link
 * Task#main(String[]) main()} method is called, passing in the task's command
 * line arguments. The task may read and take tuples from tuple space and put
 * tuples into tuple space. However, no on-demand rules will be fired when
 * finish tasks put tuples into tuple space.
 * </UL>
 * <P>
 * For further information about execution of a job, see class {@linkplain Job
 * Job}.
 *
 * @author  Alan Kaminsky
 * @version 30-Jun-2014
 */
public class Rule
	{

// Hidden data members.

	JobProperties properties;
	String[] jvmFlags;
	boolean atStart = true;
	boolean atFinish = false;
	AList<TaskSpec> taskGroup = new AList<TaskSpec>();

	// List of templates for this rule.
	AList<Tuple> templates = null;

	// Matcher for this rule.
	private Matcher matcher = new DefaultMatcher();

// Hidden constructors.

	/**
	 * Construct a new rule.
	 *
	 * @param  properties  Initial job properties.
	 * @param  jvmFlags    Initial JVM flags.
	 */
	Rule
		(JobProperties properties,
		 String[] jvmFlags)
		{
		this.properties = properties;
		this.jvmFlags = jvmFlags;
		}

// Exported operations.

	/**
	 * Specify that this rule is a start rule. If the firing condition is not
	 * specified, this rule is a start rule by default.
	 *
	 * @return  This rule.
	 */
	public Rule atStart()
		{
		atStart = true;
		atFinish = false;
		templates = null;
		return this;
		}

	/**
	 * Specify that this rule is an on-demand rule, and add the given template
	 * to this rule's list of templates.
	 *
	 * @param  template  Template.
	 *
	 * @return  This rule.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>template</TT> is null.
	 */
	public Rule whenMatch
		(Tuple template)
		{
		if (template == null)
			throw new NullPointerException
				("Rule.whenMatch(): template is null");
		atStart = false;
		atFinish = false;
		if (templates == null)
			templates = new AList<Tuple>();
		templates.addLast (template);
		return this;
		}

	/**
	 * Specify that this rule is a finish rule.
	 *
	 * @return  This rule.
	 */
	public Rule atFinish()
		{
		atStart = false;
		atFinish = true;
		templates = null;
		return this;
		}

	/**
	 * Specify the matcher for this rule. The matcher encapsulates the algorithm
	 * that matches an on-demand rule's templates against the tuples in tuple
	 * space. If not specified, the default matching algorithm is used, as
	 * described above.
	 *
	 * @param  matcher  Matcher.
	 *
	 * @return  This rule.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>matcher</TT> is null.
	 */
	public Rule matcher
		(Matcher matcher)
		{
		if (matcher == null)
			throw new NullPointerException
				("Rule.matcher(): matcher is null");
		this.matcher = matcher;
		return this;
		}

	/**
	 * Add the given task to this rule's task group. When this rule fires, one
	 * instance of the given task will be executed. Call methods on the returned
	 * {@linkplain TaskSpec} object to configure the task.
	 *
	 * @param  <T>        Task data type.
	 * @param  taskClass  Task subclass.
	 *
	 * @return  Task specification for the given task.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>taskClass</TT> is null.
	 */
	public <T extends Task> TaskSpec task
		(Class<T> taskClass)
		{
		return task (1, taskClass);
		}

	/**
	 * Add the given number of copies of the given task to this rule's task
	 * group. When this rule fires, the given number of instances of the given
	 * task will be executed. Call methods on the returned {@linkplain TaskSpec}
	 * object to configure the tasks (all copies will be configured the same).
	 *
	 * @param  <T>        Task data type.
	 * @param  copies     Number of copies (&ge; 1).
	 * @param  taskClass  Task subclass.
	 *
	 * @return  Task specification for the given task.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>copies</TT> &lt; 1.
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>taskClass</TT> is null.
	 */
	public <T extends Task> TaskSpec task
		(int copies,
		 Class<T> taskClass)
		{
		if (copies < 1)
			throw new IllegalArgumentException (String.format
				("Rule.task(): copies = %d illegal", copies));
		if (taskClass == null)
			throw new NullPointerException
				("Rule.task(): taskClass is null");
		TaskSpec taskSpec =
			new TaskSpec (this, properties, jvmFlags, taskClass);
		for (int i = 0; i < copies; ++ i)
			taskGroup.addLast (taskSpec);
		return taskSpec;
		}

	/**
	 * Determine if the tuples in tuple space match the templates in this rule.
	 * The given {@linkplain TupleSpace.Transaction Transaction} is used to find
	 * and remove matching tuples. The matching algorithm is performed by this
	 * rule's {@linkplain Rule.Matcher Matcher}.
	 *
	 * @param  transaction  Tuple space transaction.
	 *
	 * @see  #matcher(Matcher)
	 */
	public void match
		(TupleSpace.Transaction transaction)
		{
		matcher.match (transaction, templates);
		}

// Helper classes/interfaces.

	/**
	 * Interface Rule.Matcher specifies the interface for an object that matches
	 * an on-demand {@linkplain Rule}'s templates against the tuples in tuple
	 * space.
	 *
	 * @author  Alan Kaminsky
	 * @version 14-Dec-2013
	 */
	public static interface Matcher
		{
		/**
		 * Determine if the tuples in tuple space match the given templates. The
		 * templates are the firing conditions for an on-demand {@linkplain
		 * Rule}. The given {@linkplain TupleSpace.Transaction Transaction} is
		 * used to find and remove matching tuples from tuple space. If the
		 * match succeeds, the transaction must be committed; if the match
		 * fails, the transaction must be aborted.
		 * <P>
		 * <B><I>Warning:</I></B> The <TT>match()</TT> method must not alter the
		 * <TT>templates</TT> list or any of its elements.
		 *
		 * @param  transaction  Tuple space transaction.
		 * @param  templates    List of templates.
		 */
		public void match
			(TupleSpace.Transaction transaction,
			 AList<Tuple> templates);
		}

	/**
	 * Class Rule.DefaultMatcher implements the default algorithm for matching
	 * an on-demand {@linkplain Rule}'s templates against the tuples in tuple
	 * space.
	 *
	 * @author  Alan Kaminsky
	 * @version 14-Dec-2013
	 */
	private static class DefaultMatcher
		implements Matcher
		{
		/**
		 * Determine if the tuples in tuple space match the given templates. The
		 * templates are the firing conditions for an on-demand {@linkplain
		 * Rule}. The given {@linkplain TupleSpace.Transaction Transaction} is
		 * used to find and remove matching tuples from tuple space. If the
		 * match succeeds, the transaction must be committed; if the match
		 * fails, the transaction must be aborted.
		 * <P>
		 * <B><I>Warning:</I></B> The <TT>match()</TT> method must not alter the
		 * <TT>templates</TT> list or any of its elements.
		 *
		 * @param  transaction  Tuple space transaction.
		 * @param  templates    List of templates.
		 */
		public void match
			(TupleSpace.Transaction transaction,
			 AList<Tuple> templates)
			{
			// Consider each template.
			int nTemplates = templates.size();
			boolean matched = true;
			for (int i = 0; matched && i < nTemplates; ++ i)
				{
				Tuple template = templates.get (i);

				// Search tuple space for a tuple that matches current template.
				matched = false;
				Iterator<TupleSpace.TupleRef> iter = transaction.iterator();
				while (! matched && iter.hasNext())
					{
					TupleSpace.TupleRef ref = iter.next();
					Tuple target = ref.tuple;
					if (template.match (target))
						{
						// Matching tuple was found; tentatively remove it.
						transaction.remove (ref);
						matched = true;
						}
					}
				}

			// Commit/abort transaction if all templates were/were not matched.
			if (matched)
				transaction.commit();
			else
				transaction.abort();
			}
		}

	}
