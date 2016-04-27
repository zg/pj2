//******************************************************************************
//
// File:    GoAheadTuple.java
// Package: edu.rit.pjmr
// Unit:    Class edu.rit.pjmr.GoAheadTuple
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

import edu.rit.pj2.tuple.EmptyTuple;

/**
 * Class GoAheadTuple is a tuple that indicates permission for a mapper task to
 * send a {@linkplain PairTuple}.
 *
 * @author  Alan Kaminsky
 * @version 12-Jan-2015
 */
class GoAheadTuple
	extends EmptyTuple
	{
	}
