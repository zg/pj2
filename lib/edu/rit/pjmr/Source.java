//******************************************************************************
//
// File:    Source.java
// Package: edu.rit.pjmr
// Unit:    Interface edu.rit.pjmr.Source
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

import edu.rit.io.Streamable;
import edu.rit.util.Pair;
import java.io.IOException;

/**
 * Interface Source specifies the interface for a data source in the Parallel
 * Java Map Reduce Framework.
 * <P>
 * <I>Note:</I> A class that implements interface Source must be streamable.
 * <P>
 * <I>Note:</I> A class that implements interface Source must be multiple thread
 * safe.
 *
 * @param  <I>  Data type for data record ID.
 * @param  <C>  Data type for data record contents.
 *
 * @author  Alan Kaminsky
 * @version 12-Jan-2015
 */
public interface Source<I,C>
	extends Streamable
	{

// Exported operations.

	/**
	 * Open this data source.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void open()
		throws IOException;

	/**
	 * Get the next data record from this data source. The data record is being
	 * requested by the mapper with the given ID. The data record is a
	 * {@linkplain Pair} containing a data record ID and the data record
	 * contents.
	 * <P>
	 * The mapper ID is an integer &ge; 0 that uniquely identifies a particular
	 * {@link Mapper}. The source cannot assume anything about which mapper IDs
	 * will be passed to the <TT>next()</TT> method. The source is allowed to
	 * ignore the mapper ID, if it makes no difference.
	 *
	 * @param  id  Mapper ID.
	 *
	 * @return  Data record, or null if there are no more.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public Pair<I,C> next
		(int id)
		throws IOException;

	/**
	 * Close this data source.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void close()
		throws IOException;

	}
