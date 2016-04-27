//******************************************************************************
//
// File:    TextDirectorySource.java
// Package: edu.rit.pjmr
// Unit:    Class edu.rit.pjmr.TextDirectorySource
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
import java.io.FileFilter;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Scanner;
import java.util.regex.Pattern;

/**
 * Class TextDirectorySource provides a data source that obtains data records
 * from multiple files in a directory. Each file is assumed to contain plain
 * text. Each line of each file constitutes one data record. The data record ID
 * consists of the file and the line number (type {@linkplain TextId}). The data
 * record contents is the line itself (type {@linkplain java.lang.String
 * String}), excluding any line separator at the end.
 * <P>
 * Conceptually, a text directory source concatenates together all the files in
 * a given directory whose file names match a given pattern. The default is to
 * match every file in the directory. The text directory source feeds out the
 * lines of the concatenated files. The order in which the files are
 * concatenated is not specified. If multiple mapper objects are obtaining data
 * records from the same text directory source, the particular mapper object
 * that receives each data record is not specified; the lines of one file might
 * not all go to the same mapper.
 * <P>
 * Normally, it is not considered an error if the given directory does not
 * contain any files whose names match the given pattern; in this case, the
 * {@link #open() open()} method will succeed and the {@link #next(int) next()}
 * method will return null. If the lack of any matching file names <I>is</I> to
 * be considered an error, call the {@link #noMatchIsError(boolean)
 * noMatchIsError()} method after constructing the text directory source object;
 * then the {@link #open() open()} method will throw an exception in this case.
 * <P>
 * <I>Note:</I> Class TextDirectorySource is multiple thread safe.
 *
 * @author  Alan Kaminsky
 * @version 12-Jan-2015
 */
public class TextDirectorySource
	implements Source<TextId,String>
	{

// Hidden data members.

	private File directory;
	private Pattern pattern;
	private boolean noMatchIsError;
	private File[] files;
	private int fileIndex;
	private FileInputStream fis;
	private BufferedInputStream bis;
	private Scanner scanner;
	private long linenum;

// Exported constructors.

	/**
	 * Construct a new uninitialized text directory source. This constructor is
	 * for use only by object deserialization.
	 */
	public TextDirectorySource()
		{
		}

	/**
	 * Construct a new text directory source that will read all files in the
	 * given directory name.
	 *
	 * @param  directoryName  Directory name.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>directoryName</TT> is null.
	 */
	public TextDirectorySource
		(String directoryName)
		{
		if (directoryName == null)
			throw new NullPointerException
				("TextDirectorySource(): directoryName is null");
		this.directory = new File (directoryName);
		this.pattern = Pattern.compile (".*");
		}

	/**
	 * Construct a new text directory source that will read all files in the
	 * given directory.
	 *
	 * @param  directory  Directory.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>directory</TT> is null.
	 */
	public TextDirectorySource
		(File directory)
		{
		if (directory == null)
			throw new NullPointerException
				("TextDirectorySource(): directory is null");
		this.directory = directory;
		this.pattern = Pattern.compile (".*");
		}

	/**
	 * Construct a new text directory source that will read all files in the
	 * given directory name whose names match the given pattern string.
	 *
	 * @param  directoryName  Directory name.
	 * @param  patternString  Pattern string for file names, in the syntax of
	 *                        class {@linkplain java.util.regex.Pattern
	 *                        Pattern}.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>directoryName</TT> is null.
	 *     Thrown if <TT>patternString</TT> is null.
	 * @exception  PatternSyntaxException
	 *     (unchecked exception) Thrown if <TT>patternString</TT> does not obey
	 *     the legal pattern syntax.
	 */
	public TextDirectorySource
		(String directoryName,
		 String patternString)
		{
		if (directoryName == null)
			throw new NullPointerException
				("TextDirectorySource(): directoryName is null");
		if (patternString == null)
			throw new NullPointerException
				("TextDirectorySource(): patternString is null");
		this.directory = new File (directoryName);
		this.pattern = Pattern.compile (patternString);
		}

	/**
	 * Construct a new text directory source that will read all files in the
	 * given directory whose names match the given pattern string.
	 *
	 * @param  directory      Directory.
	 * @param  patternString  Pattern string for file names, in the syntax of
	 *                        class {@linkplain java.util.regex.Pattern
	 *                        Pattern}.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>directory</TT> is null. Thrown if
	 *     <TT>patternString</TT> is null.
	 * @exception  PatternSyntaxException
	 *     (unchecked exception) Thrown if <TT>patternString</TT> does not obey
	 *     the legal pattern syntax.
	 */
	public TextDirectorySource
		(File directory,
		 String patternString)
		{
		if (directory == null)
			throw new NullPointerException
				("TextDirectorySource(): directory is null");
		if (patternString == null)
			throw new NullPointerException
				("TextDirectorySource(): patternString is null");
		this.directory = directory;
		this.pattern = Pattern.compile (patternString);
		}

	/**
	 * Construct a new text directory source that will read all files in the
	 * given directory name whose names match the given pattern.
	 *
	 * @param  directoryName  Directory name.
	 * @param  pattern        Pattern for file names.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>directoryName</TT> is null.
	 *     Thrown if <TT>pattern</TT> is null.
	 */
	public TextDirectorySource
		(String directoryName,
		 Pattern pattern)
		{
		if (directoryName == null)
			throw new NullPointerException
				("TextDirectorySource(): directoryName is null");
		if (pattern == null)
			throw new NullPointerException
				("TextDirectorySource(): pattern is null");
		this.directory = new File (directoryName);
		this.pattern = pattern;
		}

	/**
	 * Construct a new text directory source that will read all files in the
	 * given directory whose names match the given pattern.
	 *
	 * @param  directory  Directory.
	 * @param  pattern    Pattern for file names.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>directory</TT> is null. Thrown if
	 *     <TT>pattern</TT> is null.
	 */
	public TextDirectorySource
		(File directory,
		 Pattern pattern)
		{
		if (directory == null)
			throw new NullPointerException
				("TextDirectorySource(): directory is null");
		if (pattern == null)
			throw new NullPointerException
				("TextDirectorySource(): pattern is null");
		this.directory = directory;
		this.pattern = pattern;
		}

// Exported operations.

	/**
	 * Open this data source.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred. Thrown if {@link
	 *     #noMatchIsError(boolean) noMatchIsError()} has been set to true and
	 *     the given directory does not contain any files whose names match the
	 *     given pattern.
	 */
	public synchronized void open()
		throws IOException
		{
		// Verify preconditions.
		if (files != null)
			throw new IOException (String.format
				("TextDirectorySource.open(): Directory %s already open",
				 directory));

		// Get a list of files to read.
		files = directory.listFiles (new FileFilter()
			{
			public boolean accept (File pathname)
				{
				return pattern.matcher (pathname.getName()) .matches();
				}
			});
		if (files == null)
			throw new IOException (String.format
				("TextDirectorySource.open(): Cannot list directory %s",
				 directory));
		if (files.length == 0 && noMatchIsError)
			throw new IOException (String.format
				("TextDirectorySource.open(): No matching files in directory %s",
				 directory));

		// Open first file.
		fileIndex = 0;
		fis = new FileInputStream (files[fileIndex]);
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
		// Repeat until we read a record or we run out of files.
		while (fileIndex < files.length)
			{
			// Read the next record from the current file.
			if (scanner.hasNextLine())
				return new Pair<TextId,String>
					(new TextId (files[fileIndex], ++ linenum),
					 scanner.nextLine());

			// Go to the next file.
			else
				{
				scanner.close();
				fis = null;
				bis = null;
				scanner = null;
				++ fileIndex;
				if (fileIndex < files.length)
					{
					fis = new FileInputStream (files[fileIndex]);
					bis = new BufferedInputStream (fis);
					scanner = new Scanner (bis);
					linenum = 0;
					}
				}
			}

		// No more files.
		return null;
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
		if (fileIndex < files.length)
			{
			scanner.close();
			fis = null;
			bis = null;
			scanner = null;
			fileIndex = files.length;
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
		out.writeObject (directory);
		out.writeObject (pattern);
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
		directory = (File) in.readObject();
		pattern = (Pattern) in.readObject();
		}

	/**
	 * Specify whether the lack of any files whose names match the given pattern
	 * is to be considered an error. The default is false.
	 *
	 * @param  err  True if this situation is an error, false if it isn't.
	 *
	 * @return  This text directory source object.
	 */
	public synchronized TextDirectorySource noMatchIsError
		(boolean err)
		{
		this.noMatchIsError = err;
		return this;
		}

	}
