//******************************************************************************
//
// File:    TextFileSource.java
// Package: edu.rit.pjmr
// Unit:    Class edu.rit.pjmr.TextFileSource
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

import edu.rit.io.InStream;
import edu.rit.io.OutStream;
import edu.rit.util.Pair;
import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Scanner;

/**
 * Class TextFileSource provides a data source that obtains data records from a
 * file. The file is assumed to contain plain text. Each line of the file
 * constitutes one data record. The data record ID consists of the file and the
 * line number (type {@linkplain TextId}). The data record contents is the line
 * itself (type {@linkplain java.lang.String String}), excluding any line
 * separator at the end.
 * <P>
 * <I>Note:</I> Class TextFileSource is multiple thread safe.
 *
 * @author  Alan Kaminsky
 * @version 12-Jan-2015
 */
public class TextFileSource
	implements Source<TextId,String>
	{

// Hidden data members.

	private File file;
	private FileInputStream fis;
	private BufferedInputStream bis;
	private Scanner scanner;
	private long linenum;

// Exported constructors.

	/**
	 * Construct a new uninitialized text file source. This constructor is for
	 * use only by object deserialization.
	 */
	public TextFileSource()
		{
		}

	/**
	 * Construct a new text file source that will read the given file name.
	 *
	 * @param  filename  Text file name.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>file</TT> is null.
	 */
	public TextFileSource
		(String filename)
		{
		if (filename == null)
			throw new NullPointerException
				("TextFileSource(): filename is null");
		this.file = new File (filename);
		}

	/**
	 * Construct a new text file source that will read the given file.
	 *
	 * @param  file  Text file.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>file</TT> is null.
	 */
	public TextFileSource
		(File file)
		{
		if (file == null)
			throw new NullPointerException
				("TextFileSource(): file is null");
		this.file = file;
		}

// Exported operations.

	/**
	 * Open this data source.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public synchronized void open()
		throws IOException
		{
		if (fis != null)
			throw new IOException (String.format
				("TextFileSource.open(): File %s already open", file));
		fis = new FileInputStream (file);
		bis = new BufferedInputStream (fis);
		scanner = new Scanner (bis);
		linenum = 0;
		}

	/**
	 * Get the next data record from this data source. The data record is being
	 * requested by the mapper with the given ID. The data record is a
	 * {@linkplain Pair} containing a data record ID and the data record
	 * contents.
	 *
	 * @param  id  Mapper ID (ignored).
	 *
	 * @return  Data record, or null if there are no more.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public synchronized Pair<TextId,String> next
		(int id)
		throws IOException
		{
		if (scanner == null)
			throw new IOException (String.format
				("TextFileSource.next(): File %s not open", file));
		return scanner.hasNextLine() ?
			new Pair<TextId,String>
				(new TextId (file, ++ linenum),
				 scanner.nextLine()) :
			null;
		}

	/**
	 * Close this data source.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public synchronized void close()
		throws IOException
		{
		if (scanner == null)
			throw new IOException (String.format
				("TextFileSource.close(): File %s not open", file));
		try
			{
			scanner.close();
			}
		finally
			{
			fis = null;
			bis = null;
			scanner = null;
			}
		}

	/**
	 * Write this data source to the given out stream.
	 *
	 * @param  out  Out stream.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public synchronized void writeOut
		(OutStream out)
		throws IOException
		{
		out.writeObject (file);
		}

	/**
	 * Read this data source from the given in stream.
	 *
	 * @param  in  In stream.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public synchronized void readIn
		(InStream in)
		throws IOException
		{
		file = (File) in.readObject();
		}

	}
