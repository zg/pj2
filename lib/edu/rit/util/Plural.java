//******************************************************************************
//
// File:    Plural.java
// Package: edu.rit.util
// Unit:    Class edu.rit.util.Plural
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

package edu.rit.util;

/**
 * Class Plural provides an object for printing a number followed by a word with
 * the proper pluralization.
 *
 * @author  Alan Kaminsky
 * @version 14-Nov-2011
 */
public class Plural
	{

// Hidden data members.

	private String s;

// Exported constructors.

	/**
	 * Construct a new plural object. If the number is 1, the string version is
	 * <TT>"&lt;number&gt; &lt;word&gt;"</TT>. Otherwise, the string version is
	 * <TT>"&lt;number&gt; &lt;word&gt;s"</TT>.
	 *
	 * @param  number  Number to print.
	 * @param  word    Word to print.
	 */
	public Plural
		(int number,
		 String word)
		{
		if (Math.abs (number) == 1) s = number + " " + word;
		else s = number + " " + word + "s";
		}

	/**
	 * Construct a new plural object. If the number is 1, the string version is
	 * <TT>"&lt;number&gt; &lt;word&gt;"</TT>. Otherwise, the string version is
	 * <TT>"&lt;number&gt; &lt;word&gt;s"</TT>.
	 *
	 * @param  number  Number to print.
	 * @param  word    Word to print.
	 */
	public Plural
		(long number,
		 String word)
		{
		if (Math.abs (number) == 1L) s = number + " " + word;
		else s = number + " " + word + "s";
		}

	/**
	 * Construct a new plural object. If the number is 1, the string version is
	 * <TT>"&lt;number&gt; &lt;singularWord&gt;"</TT>. Otherwise, the string
	 * version is <TT>"&lt;number&gt; &lt;pluralWord&gt;"</TT>.
	 *
	 * @param  number        Number to print.
	 * @param  singularWord  Singular word to print.
	 * @param  pluralWord    Plural word to print.
	 */
	public Plural
		(int number,
		 String singularWord,
		 String pluralWord)
		{
		if (Math.abs (number) == 1) s = number + " " + singularWord;
		else s = number + " " + pluralWord;
		}

	/**
	 * Construct a new plural object. If the number is 1, the string version is
	 * <TT>"&lt;number&gt; &lt;singularWord&gt;"</TT>. Otherwise, the string
	 * version is <TT>"&lt;number&gt; &lt;pluralWord&gt;"</TT>.
	 *
	 * @param  number        Number to print.
	 * @param  singularWord  Singular word to print.
	 * @param  pluralWord    Plural word to print.
	 */
	public Plural
		(long number,
		 String singularWord,
		 String pluralWord)
		{
		if (Math.abs (number) == 1) s = number + " " + singularWord;
		else s = number + " " + pluralWord;
		}

// Exported operations.

	/**
	 * Returns a string version of this plural object.
	 *
	 * @return  String version.
	 */
	public String toString()
		{
		return s;
		}

	}
