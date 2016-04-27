//******************************************************************************
//
// File:    FileResource.java
// Package: edu.rit.io
// Unit:    Class edu.rit.io.FileResource
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

package edu.rit.io;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.File;
import java.net.URL;

/**
 * Class FileResource encapsulates a file stored as a resource in the Java class
 * path.
 *
 * @author  Alan Kaminsky
 * @version 19-Feb-2014
 */
public class FileResource
	{

// Hidden data members.

	private File file;

// Exported constructors.

	/**
	 * Construct a new file resource.
	 *
	 * @param  name  Resource name relative to the Java class path.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public FileResource
		(String name)
		throws IOException
		{
		// Get URL for resource.
		URL url = Thread.currentThread().getContextClassLoader()
			.getResource (name);
		if (url == null)
			throw new FileNotFoundException (String.format
				("FileResource(): Resource \"%s\" not found", name));

		// For a file: URL, use the file itself.
		if (url.getProtocol().equals ("file"))
			{
			file = new File (url.getPath());
			}

		// For any other URL, copy the resource into a temporary file, which
		// will be deleted when the JVM exits.
		else
			{
			name = new File (name) .getName();
			int i = name.lastIndexOf ('.');
			String fprefix = i == -1 ? name : name.substring (0, i);
			String fsuffix = i == -1 ? null : name.substring (i);
			file = File.createTempFile (fprefix+"_tmp", fsuffix);
			OutputStream out =
				new BufferedOutputStream
					(new FileOutputStream (file));
			InputStream in =
				new BufferedInputStream
					(url.openStream());
			while ((i = in.read()) != -1) out.write (i);
			out.close();
			in.close();
			file.deleteOnExit();
			}
		}

// Exported operations.

	/**
	 * Get the file for reading the file resource.
	 *
	 * @return  File.
	 */
	public File file()
		{
		return file;
		}

	/**
	 * Get the file name for reading the file resource.
	 *
	 * @return  File name.
	 */
	public String filename()
		{
		return file.getPath();
		}

	}
