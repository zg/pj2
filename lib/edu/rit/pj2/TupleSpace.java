//******************************************************************************
//
// File:    TupleSpace.java
// Package: edu.rit.pj2
// Unit:    Class edu.rit.pj2.TupleSpace
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

import edu.rit.util.AList;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Class TupleSpace provides <I>tuple space</I> for a PJ2 {@linkplain Job Job}.
 * Tuple space is the repository for the {@linkplain Tuple Tuple}s that the
 * {@linkplain Job Job}'s {@linkplain Task Task}s use to communicate and
 * coordinate with each other.
 * <P>
 * Normally, user programs do not call methods directly on a tuple space object.
 * However, a class implementing interface {@linkplain Rule.Matcher
 * Rule.Matcher} does call methods on a {@linkplain Transaction Transaction}
 * object to find target tuples that match the rule's template tuples and remove
 * the matching tuples from tuple space.
 * <P>
 * To put a tuple into tuple space, call the {@link #putTuple(Tuple) putTuple()}
 * method. 
 * <P>
 * To remove tuple(s) from tuple space, call the {@link #getTransaction()
 * getTransaction()} method to get a {@linkplain Transaction Transaction}
 * object, then use the transaction object to find and remove the desired
 * tuple(s).
 * <P>
 * Tuple space is locked while a tuple is being put. Tuple space is locked while
 * a transaction is in progress, from when the transaction is created until when
 * the transaction is committed or aborted. If tuple space is locked, the {@link
 * #putTuple(Tuple) putTuple()} and {@link #getTransaction() getTransaction()}
 * methods will block until tuple space is unlocked.
 *
 * @author  Alan Kaminsky
 * @version 30-Jun-2014
 */
public class TupleSpace
	{

// Tuple reference.

	/**
	 * Class TupleSpace.TupleRef provides a reference to a tuple in tuple space.
	 * <P>
	 * The TupleRef class is needed because the same tuple object may be put
	 * into tuple space more than once. In that case tuple space will contain
	 * multiple tuple references referring to the same tuple object.
	 *
	 * @author  Alan Kaminsky
	 * @version 24-Oct-2013
	 */
	public static class TupleRef
		{
		/**
		 * Reference to the tuple.
		 */
		public final Tuple tuple;

		// True if this tuple reference has been tentatively removed.
		private boolean removed;

		// Construct a new tuple reference to the given tuple.
		private TupleRef
			(Tuple tuple)
			{
			this.tuple = tuple;
			}
		}

// Hidden data members.

	private ReentrantLock lock = new ReentrantLock();

	// List (set) of tuple references.
	private LinkedHashSet<TupleRef> tupleList = new LinkedHashSet<TupleRef>();

// Exported constructors.

	/**
	 * Construct a new tuple space.
	 */
	public TupleSpace()
		{
		}

// Exported operations.

	/**
	 * Put the given tuple into this tuple space. The <TT>putTuple()</TT> method
	 * will block until no other <TT>putTuple()</TT> method call is in progress
	 * and no transaction is in progress.
	 *
	 * @param  tuple  Tuple.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>tuple</TT> is null.
	 */
	public void putTuple
		(Tuple tuple)
		{
		if (tuple == null)
			throw new NullPointerException
				("TupleSpace.putTuple(): tuple is null");

		lock.lock();

		TupleRef ref = new TupleRef (tuple);
		tupleList.add (ref);

		lock.unlock();
		}

	/**
	 * Get a transaction for finding and removing tuples from this tuple space.
	 * The <TT>getTransaction()</TT> method will block until no
	 * <TT>putTuple()</TT> method call is in progress and no other transaction
	 * is in progress.
	 *
	 * @return  Transaction.
	 */
	public Transaction getTransaction()
		{
		lock.lock();
		return new Transaction();
		}

	/**
	 * Get a list of all tuples in tuple space.
	 *
	 * @return  List of all tuples.
	 */
	public AList<Tuple> allTuples()
		{
		lock.lock();

		AList<Tuple> list = new AList<Tuple>();
		for (TupleRef ref : tupleList)
			list.addLast (ref.tuple);

		lock.unlock();
		return list;
		}

	/**
	 * Remove all tuples from tuple space.
	 */
	public void clear()
		{
		lock.lock();
		tupleList.clear();
		lock.unlock();
		}

// Transaction class.

	/**
	 * Class TupleSpace.Transaction provides a transaction for finding and
	 * removing {@linkplain Tuple Tuple}s from {@linkplain TupleSpace
	 * TupleSpace}. To perform a transaction:
	 * <OL TYPE=1>
	 * <P><LI>
	 * Create a transaction by calling the {@link TupleSpace#getTransaction()
	 * getTransaction()} method on the {@linkplain TupleSpace TupleSpace}
	 * object. At this point the tuple space object becomes locked.
	 * <P><LI>
	 * Call the transaction's {@link #iterator() iterator()} method to get
	 * an iterator for scanning the references to tuples in tuple space.
	 * <P><LI>
	 * Use the iterator's methods to scan the tuples.
	 * <P><LI>
	 * To remove a certain tuple from tuple space, call the transaction's {@link
	 * #remove(TupleSpace.TupleRef) remove()} method. This <I>tentatively</I>
	 * removes the tuple. (Don't call the iterator's <TT>remove()</TT> method;
	 * that method is not supported.)
	 * <P><LI>
	 * To put a removed tuple back into tuple space, call the transaction's
	 * {@link #restore(TupleSpace.TupleRef) restore()} method. This
	 * <I>tentatively</I> restores the tuple.
	 * <P><LI>
	 * Steps 2, 3, 4, and 5 may be repeated as many times as necessary.
	 * <P><LI>
	 * To <I>permanently</I> accept all tentative changes made during the
	 * transaction, call the transaction's {@link #commit() commit()} method. At
	 * this point the transaction is finished and the tuple space object becomes
	 * unlocked.
	 * <P><LI>
	 * To undo all tentative changes and put tuple space back the way it was
	 * before the transaction was created, call the transaction's {@link
	 * #abort() abort()} method. At this point the transaction is finished and
	 * the tuple space object becomes unlocked.
	 * <P><LI>
	 * Call the transaction's {@link #getTuples() getTuples()} method, which
	 * returns a list of the tuples that were permanently removed by the
	 * transaction (if any).
	 * </OL>
	 * <P>
	 * Class TupleSpace.Transaction is not multiple thread safe. It is assumed
	 * that the same thread will create a transaction and call the transaction's
	 * methods until the transaction is committed or aborted.
	 *
	 * @author  Alan Kaminsky
	 * @version 30-Jun-2014
	 */
	public class Transaction
		{
		private boolean inProgress = true;
		private LinkedHashSet<TupleRef> removedTupleRefs =
			new LinkedHashSet<TupleRef>();

		private Transaction()
			{
			}

		/**
		 * Get an iterator for scanning the references to tuples in tuple space.
		 * <P>
		 * <I>Note:</I> The returned iterator's <TT>remove()</TT> method is not
		 * supported.
		 *
		 * @return  Iterator.
		 *
		 * @exception  IllegalStateException
		 *     (unchecked exception) Thrown if this transaction is not in
		 *     progress.
		 */
		public Iterator<TupleRef> iterator()
			{
			if (! inProgress)
				throw new IllegalStateException
					("TupleSpace.Transaction.iterator(): Transaction not in progress");

			return tupleList.iterator();
			}

		/**
		 * Tentatively remove the given tuple reference from tuple space.
		 *
		 * @param  ref  Tuple reference.
		 *
		 * @exception  IllegalStateException
		 *     (unchecked exception) Thrown if this transaction is not in
		 *     progress.
		 * @exception  NullPointerException
		 *     (unchecked exception) Thrown if <TT>ref</TT> is null.
		 * @exception  IllegalArgumentException
		 *     (unchecked exception) Thrown if <TT>ref</TT> does not exist in
		 *     tuple space.
		 */
		public void remove
			(TupleRef ref)
			{
			if (! inProgress)
				throw new IllegalStateException
					("TupleSpace.Transaction.remove(): Transaction not in progress");
			if (ref == null)
				throw new NullPointerException
					("TupleSpace.Transaction.remove(): ref is null");

			if (! tupleList.contains (ref))
				throw new IllegalArgumentException
					("TupleSpace.Transaction.remove(): ref not in tuple space");

			ref.removed = true;
			removedTupleRefs.add (ref);
			}

		/**
		 * Tentatively restore the given tuple reference to tuple space.
		 *
		 * @param  ref  Tuple reference.
		 *
		 * @exception  IllegalStateException
		 *     (unchecked exception) Thrown if this transaction is not in
		 *     progress.
		 * @exception  NullPointerException
		 *     (unchecked exception) Thrown if <TT>ref</TT> is null.
		 * @exception  IllegalArgumentException
		 *     (unchecked exception) Thrown if <TT>ref</TT> does not exist in
		 *     tuple space.
		 */
		public void restore
			(TupleRef ref)
			{
			if (! inProgress)
				throw new IllegalStateException
					("TupleSpace.Transaction.restore(): Transaction not in progress");
			if (ref == null)
				throw new NullPointerException
					("TupleSpace.Transaction.restore(): ref is null");

			if (! tupleList.contains (ref))
				throw new IllegalArgumentException
					("TupleSpace.Transaction.restore(): ref not in tuple space");

			ref.removed = false;
			removedTupleRefs.remove (ref);
			}

		/**
		 * Commit this transaction. All tentative changes to tuple space become
		 * permanent. Afterwards, this transaction is no longer in progress.
		 *
		 * @exception  IllegalStateException
		 *     (unchecked exception) Thrown if this transaction is not in
		 *     progress.
		 */
		public void commit()
			{
			if (! inProgress)
				throw new IllegalStateException
					("TupleSpace.Transaction.commit(): Transaction not in progress");
			for (TupleRef ref : removedTupleRefs)
				tupleList.remove (ref);

			inProgress = false;
			lock.unlock();
			}

		/**
		 * Abort this transaction. All tentative changes to tuple space are
		 * undone. Afterwards, this transaction is no longer in progress.
		 *
		 * @exception  IllegalStateException
		 *     (unchecked exception) Thrown if this transaction is not in
		 *     progress.
		 */
		public void abort()
			{
			if (! inProgress)
				throw new IllegalStateException
					("TupleSpace.Transaction.abort(): Transaction not in progress");

			for (TupleRef ref : removedTupleRefs)
				ref.removed = false;
			removedTupleRefs.clear();

			inProgress = false;
			lock.unlock();
			}

		/**
		 * Get the tuples that were removed from tuple space during this
		 * transaction. The tuples appear in the list in the order in which the
		 * tuples were removed from tuple space. If no tuples were removed, or
		 * if this transaction was aborted, an empty list is returned.
		 *
		 * @return  List of removed tuples.
		 *
		 * @exception  IllegalStateException
		 *     (unchecked exception) Thrown if this transaction is in progress.
		 */
		public AList<Tuple> getTuples()
			{
			if (inProgress)
				throw new IllegalStateException
					("TupleSpace.Transaction.getTuples(): Transaction in progress");

			AList<Tuple> list = new AList<Tuple>();
			for (TupleRef ref : removedTupleRefs)
				list.addLast (ref.tuple);
			return list;
			}
		}

	}
