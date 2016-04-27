//******************************************************************************
//
// File:    Tuple.java
// Package: edu.rit.pj2
// Unit:    Class edu.rit.pj2.Tuple
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

package edu.rit.pj2;

import edu.rit.io.InStream;
import edu.rit.io.OutStream;
import edu.rit.io.Streamable;
import java.io.IOException;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.lang.reflect.Array;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;

/**
 * Class Tuple is the base class for a tuple. Tuples are used to coordinate and
 * communicate among multiple {@linkplain Task}s in a Parallel Java 2
 * {@linkplain Job}. A tuple's fields carry content. The Tuple base class does
 * not declare any fields; a subclass of class Tuple declares
 * application-specific fields.
 * <P>
 * <B>Immutability.</B>
 * Every tuple object is assumed to be immutable once it has been constructed
 * and initialized. "Immutable" means the states of the tuple's fields do not
 * change; the states of all objects referenced by the tuple do not change; the
 * states of all objects referenced by all objects referenced by the tuple do
 * not change; and so on. If you need to change the contents of a tuple or any
 * object to which the tuple refers (directly or indirectly), make a <I>deep
 * copy</I> of the tuple and change the copy.
 * <P>
 * <B>Cloning.</B>
 * Tuples are cloneable. The {@link #clone() clone()} method must make a <I>deep
 * copy</I> of the tuple. The base class Tuple's {@link #clone() clone()} method
 * creates a <I>shallow copy</I> of the tuple. A subclass of class Tuple can use
 * the inherited {@link #clone() clone()} method if all the subclass's fields
 * are of primitive types; in that case, a shallow copy is the same as a deep
 * copy. If some of the subclass's fields are objects or arrays, the subclass
 * must override the {@link #clone() clone()} method to create a <I>deep
 * copy</I> of the tuple.
 * <P>
 * <B>Object streaming.</B>
 * Tuples are streamable. A subclass of class Tuple must override the {@link
 * #writeOut(OutStream) writeOut()} and {@link #readIn(InStream) readIn()}
 * methods to write and read the subclass's fields. Also, a subclass of class
 * Tuple must have a no-argument constructor; this constructor need not be
 * public. If a tuple subclass does not provide the {@link #writeOut(OutStream)
 * writeOut()} method, the {@link #readIn(InStream) readIn()} method, and the
 * no-argument constructor, an exception will be thrown when an instance of that
 * tuple subclass is read or written.
 * <P>
 * <B>Tuple matching.</B>
 * One tuple can be <I>matched</I> with another tuple using the {@link
 * #match(Tuple) match()} method. The tuple upon which the {@link #match(Tuple)
 * match()} method is called is the <I>template.</I> The tuple supplied as the
 * {@link #match(Tuple) match()} method's argument is the <I>target.</I> The
 * {@link #match(Tuple) match()} method succeeds and returns true if both of the
 * following are true:
 * <UL>
 * <P><LI>
 * The target is an instance of the class returned by the template's {@link
 * #matchClass() matchClass()} method or a subclass thereof.
 * <P><LI>
 * The template's {@link #matchContent(Tuple) matchContent()} method returns
 * true when the target is passed as the argument.
 * </UL>
 * <P>
 * In the Tuple base class implementation, the {@link #matchClass()
 * matchClass()} method returns the template's runtime class, and the {@link
 * #matchContent(Tuple) matchContent()} method returns true. Thus, the target
 * will match the template merely if the target is an instance of the template's
 * class or a subclass thereof.
 * <P>
 * The matching algorithm for a subclass of class Tuple can be altered by
 * overriding the {@link #matchClass() matchClass()} and/or {@link
 * #matchContent(Tuple) matchContent()} methods. A typical situation is when you
 * want to match some fields of the template and target, and possibly ignore
 * other fields.
 *
 * @author  Alan Kaminsky
 * @version 07-Jan-2015
 */
public abstract class Tuple
	implements Cloneable, Streamable
	{

// Exported constructors.

	/**
	 * Construct a new tuple.
	 */
	public Tuple()
		{
		}

// Exported operations.

	/**
	 * Clone this tuple.
	 * <P>
	 * The Tuple base class's <TT>clone()</TT> method returns a <I>shallow
	 * copy</I> of this tuple.
	 *
	 * @return  Clone of this tuple.
	 */
	public Object clone()
		{
		try
			{
			return super.clone();
			}
		catch (CloneNotSupportedException exc)
			{
			throw new TerminateException ("Shouldn't happen", exc);
			}
		}

	/**
	 * Match this template tuple with the given target tuple. The
	 * <TT>match()</TT> method succeeds and returns true if both of the
	 * following are true: the target is an instance of the class returned by
	 * the {@link #matchClass() matchClass()} method or a subclass thereof; and
	 * the {@link #matchContent(Tuple) matchContent()} method returns true when
	 * the target is passed as the argument.
	 *
	 * @param  target  Target tuple.
	 *
	 * @return  True if the tuples match, false if they don't.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>target</TT> is null.
	 */
	public final boolean match
		(Tuple target)
		{
		if (target == null)
			throw new NullPointerException
				("Tuple.match(): target is null");

		return
			matchClass().isAssignableFrom (target.getClass()) &&
			matchContent (target);
		}

	/**
	 * Get this template tuple's matching class. In order to match this template
	 * tuple, a target tuple must be an instance of the returned class or a
	 * subclass thereof.
	 * <P>
	 * The Tuple base class's <TT>matchClass()</TT> method returns the runtime
	 * class of this template tuple. If desired, a subclass can override the
	 * <TT>matchClass()</TT> method to return some other class.
	 *
	 * @return  Matching class.
	 */
	public <T extends Tuple> Class<T> matchClass()
		{
		return (Class<T>) getClass();
		}

	/**
	 * Determine if the given target tuple's content matches this template's
	 * content. The target tuple is assumed to be an instance of this template's
	 * matching class or a subclass thereof.
	 * <P>
	 * The Tuple base class's <TT>matchContent()</TT> method simply returns
	 * true. If desired, a subclass can override the <TT>matchContent()</TT>
	 * method to check the fields of this template tuple and the target tuple.
	 *
	 * @param  target  Target tuple.
	 *
	 * @return  True if the target tuple's content matches this template tuple's
	 *          content, false otherwise.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>target</TT> is null.
	 */
	public boolean matchContent
		(Tuple target)
		{
		if (target == null)
			throw new NullPointerException
				("Tuple.matchContent(): target is null");

		return true;
		}

	/**
	 * Write this tuple's fields to the given out stream.
	 *
	 * @param  out  Out stream.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public abstract void writeOut
		(OutStream out)
		throws IOException;

	/**
	 * Read this tuple's fields from the given in stream.
	 *
	 * @param  in  In stream.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public abstract void readIn
		(InStream in)
		throws IOException;

	/**
	 * Dump this tuple on the standard output stream. The first line is the
	 * name of this tuple's class. Each of the remaining lines consists of the
	 * name of a public non-static field, an equals sign, and the value of the
	 * field. This method is intended for debugging.
	 */
	public void dump()
		{
		dump (System.out);
		}

	/**
	 * Dump this tuple on the given print stream. The first line is the name of
	 * this tuple's class. Each of the remaining lines consists of the name of a
	 * public non-static field, an equals sign, and the value of the field.
	 * This method is intended for debugging.
	 *
	 * @param  out  Print stream.
	 */
	public void dump
		(PrintStream out)
		{
		dump (out, 1);
		}

	/**
	 * Dump this tuple on the given print stream at the given indent level. The
	 * first line is the name of this tuple's class. Each of the remaining lines
	 * consists of the name of a public non-static field, an equals sign, and
	 * the value of the field. This method is intended for debugging.
	 *
	 * @param  out     Print stream.
	 * @param  indent  Indent level for fields. (The class name is not
	 *                 indented.)
	 */
	public void dump
		(PrintStream out,
		 int indent)
		{
		Class<?> tupleClass = this.getClass();
		out.printf ("%s%n", tupleClass.getName());
		for (Field field : tupleClass.getFields())
			{
			field.setAccessible (true);
			if (Modifier.isStatic (field.getModifiers()))
				continue;
			try
				{
				dumpIndent (out, indent);
				out.printf ("%s = ", field.getName());
				dumpValue (field.get (this), out, indent);
				}
			catch (IllegalAccessException exc)
				{
				throw new IllegalStateException
					("Tuple.dump(): Shouldn't happen", exc);
				}
			}
		}

	/**
	 * Dump spaces for the given indent level on the given print stream.
	 *
	 * @param  out     Print stream.
	 * @param  indent  Indent level.
	 */
	private void dumpIndent
		(PrintStream out,
		 int indent)
		{
		for (int i = 0; i < indent; ++ i)
			out.printf ("   ");
		}

	/**
	 * Dump the given field value on the given print stream at the given indent
	 * level.
	 *
	 * @param  value   Field value.
	 * @param  out     Print stream.
	 * @param  indent  Indent level.
	 */
	private void dumpValue
		(Object value,
		 PrintStream out,
		 int indent)
		{
		if (value == null)
			{
			out.printf ("null%n");
			}
		else if (value instanceof Tuple)
			{
			((Tuple)value).dump (out, indent + 1);
			}
		else if (value.getClass().isArray())
			{
			out.printf ("array%n");
			dumpArray (value, out, indent + 1);
			}
		else
			{
			out.printf ("%s%n", value);
			}
		}

	/**
	 * Dump the elements of the given array on the given print stream at the
	 * given indent level.
	 *
	 * @param  array   Array.
	 * @param  out     Print stream.
	 * @param  indent  Indent level.
	 */
	private void dumpArray
		(Object array,
		 PrintStream out,
		 int indent)
		{
		int n = Array.getLength (array);
		for (int i = 0; i < n; ++ i)
			{
			dumpIndent (out, indent);
			out.printf ("[%d] = ", i);
			dumpValue (Array.get (array, i), out, indent);
			}
		}

	/**
	 * Dump this tuple on the given print writer. The first line is the name of
	 * this tuple's class. Each of the remaining lines consists of the name of a
	 * public non-static field, an equals sign, and the value of the field.
	 * This method is intended for debugging.
	 *
	 * @param  out  Print writer.
	 */
	public void dump
		(PrintWriter out)
		{
		dump (out, 1);
		}

	/**
	 * Dump this tuple on the given print writer at the given indent level. The
	 * first line is the name of this tuple's class. Each of the remaining lines
	 * consists of the name of a public non-static field, an equals sign, and
	 * the value of the field. This method is intended for debugging.
	 *
	 * @param  out     Print writer.
	 * @param  indent  Indent level for fields. (The class name is not
	 *                 indented.)
	 */
	public void dump
		(PrintWriter out,
		 int indent)
		{
		Class<?> tupleClass = this.getClass();
		out.printf ("%s%n", tupleClass.getName());
		for (Field field : tupleClass.getFields())
			{
			field.setAccessible (true);
			if (Modifier.isStatic (field.getModifiers()))
				continue;
			try
				{
				dumpIndent (out, indent);
				out.printf ("%s = ", field.getName());
				dumpValue (field.get (this), out, indent);
				}
			catch (IllegalAccessException exc)
				{
				throw new IllegalStateException
					("Tuple.dump(): Shouldn't happen", exc);
				}
			}
		}

	/**
	 * Dump spaces for the given indent level on the given print writer.
	 *
	 * @param  out     Print writer.
	 * @param  indent  Indent level.
	 */
	private void dumpIndent
		(PrintWriter out,
		 int indent)
		{
		for (int i = 0; i < indent; ++ i)
			out.printf ("   ");
		}

	/**
	 * Dump the given field value on the given print writer at the given indent
	 * level.
	 *
	 * @param  value   Field value.
	 * @param  out     Print writer.
	 * @param  indent  Indent level.
	 */
	private void dumpValue
		(Object value,
		 PrintWriter out,
		 int indent)
		{
		if (value == null)
			{
			out.printf ("null%n");
			}
		else if (value instanceof Tuple)
			{
			((Tuple)value).dump (out, indent + 1);
			}
		else if (value.getClass().isArray())
			{
			out.printf ("array%n");
			dumpArray (value, out, indent + 1);
			}
		else
			{
			out.printf ("%s%n", value);
			}
		}

	/**
	 * Dump the elements of the given array on the given print writer at the
	 * given indent level.
	 *
	 * @param  array   Array.
	 * @param  out     Print writer.
	 * @param  indent  Indent level.
	 */
	private void dumpArray
		(Object array,
		 PrintWriter out,
		 int indent)
		{
		int n = Array.getLength (array);
		for (int i = 0; i < n; ++ i)
			{
			dumpIndent (out, indent);
			out.printf ("[%d] = ", i);
			dumpValue (Array.get (array, i), out, indent);
			}
		}

	}
