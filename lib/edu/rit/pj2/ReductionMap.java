//******************************************************************************
//
// File:    ReductionMap.java
// Package: edu.rit.pj2
// Unit:    Class edu.rit.pj2.ReductionMap
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

/**
 * Class ReductionMap keeps track of shared global reduction variables and their
 * thread-local copies.
 *
 * @author  Alan Kaminsky
 * @version 21-May-2013
 */
class ReductionMap
	{

// Hidden helper classes.

	private static class Variable
		{
		public final Vbl globalVbl;
		public final Vbl localVbl;

		public Variable
			(Vbl globalVbl,
			 Vbl localVbl)
			{
			this.globalVbl = globalVbl;
			this.localVbl = localVbl;
			}

		public boolean equals
			(Object obj)
			{
			return
				(obj instanceof Variable) &&
				(((Variable)obj).globalVbl == this.globalVbl);
			}
		}

// Hidden data members.

	private AList<Variable> variables = new AList<Variable>();

// Exported constructors.

	/**
	 * Construct a new reduction map.
	 */
	public ReductionMap()
		{
		}

// Exported operations.

	/**
	 * Clear this reduction map.
	 */
	public void clear()
		{
		variables.clear();
		}

	/**
	 * Add the given global variable to this reduction map.
	 *
	 * @param  globalVbl  Global variable.
	 *
	 * @return  Thread-local copy of the global variable.
	 */
	public Vbl add
		(Vbl globalVbl)
		{
		Vbl localVbl = (Vbl) globalVbl.clone();
		variables.addLast (new Variable (globalVbl, localVbl));
		return localVbl;
		}

	/**
	 * Reduce the thread-local variables in the given reduction map into the
	 * thread-local variables in this reduction map.
	 *
	 * @param  map  Reduction map.
	 *
	 * @exception  IllegalStateException
	 *     (unchecked exception) Thrown if the given <TT>map</TT> does not
	 *     contain the same global variables in the same order as this reduction
	 *     map.
	 */
	public void reduce
		(ReductionMap map)
		{
		int n = this.variables.size();
		for (int i = 0; i < n; ++ i)
			{
			Variable thisVbl = this.variables.get (i);
			int p = map.variables.position (thisVbl);
			if (p >= 0)
				thisVbl.localVbl.reduce (map.variables.get(p).localVbl);
			}
		}

	/**
	 * Set the global variables to their corresponding thread-local variables in
	 * this reduction map.
	 */
	public void setGlobalVariables()
		{
		int n = this.variables.size();
		for (int i = 0; i < n; ++ i)
			{
			Variable vbl = variables.get (i);
			vbl.globalVbl.set (vbl.localVbl);
			}
		}

	}
