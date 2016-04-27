//******************************************************************************
//
// File:    Arrow.java
// Package: edu.rit.draw.item
// Unit:    Class edu.rit.draw.item.Arrow
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

package edu.rit.draw.item;

import java.awt.Graphics2D;
import java.awt.Shape;

import java.awt.geom.AffineTransform;
import java.awt.geom.GeneralPath;

import java.io.Externalizable;
import java.io.IOException;
import java.io.InvalidObjectException;
import java.io.ObjectInput;
import java.io.ObjectOutput;

/**
 * Class Arrow provides various kinds of arrowheads that can be added to the
 * ends of a {@linkplain Line}.
 *
 * @author  Alan Kaminsky
 * @version 06-Aug-2013
 */
public class Arrow
	implements Externalizable
	{

// Hidden data members.

	private static final long serialVersionUID = 7782719680651946887L;

	// Arrow kinds.
	private static final int MIN_KIND = 0;
	private static final int NONE_KIND = 0;
	private static final int SOLID_FIXED_KIND = 1;
	private static final int OPEN_FIXED_KIND = 2;
	private static final int SOLID_KIND = 3;
	private static final int OPEN_KIND = 4;
	private static final int MAX_KIND = 4;
		// Note: SOLID_FIXED_KIND and OPEN_FIXED_KIND use a fixed arrow width of
		// 6.0, for compatibility with older versions of class Arrow.

	// For drawing arrows.
	private static final float GOLDEN_RATIO = (float)
		((1.0 + Math.sqrt (5.0)) / 2.0);

	// Fixed arrow size.
	private static final float FIXED_SIZE = 6.0f;

	// Default arrow size.
	private static float theDefaultSize = FIXED_SIZE;

	// This arrow's kind and size.
	private int myKind;
	private float mySize;

// Exported constants.

	/**
	 * A nonexistent arrow. The arrow size is 6.0.
	 */
	public static final Arrow NONE = new Arrow (NONE_KIND, FIXED_SIZE);

	/**
	 * A solid arrow in the shape of a narrow triangle. The arrow size is 6.0.
	 */
	public static final Arrow SOLID = new Arrow (SOLID_FIXED_KIND, FIXED_SIZE);

	/**
	 * An open arrow in the shape of a narrow triangle. The arrow size is 6.0.
	 */
	public static final Arrow OPEN = new Arrow (OPEN_FIXED_KIND, FIXED_SIZE);

// Exported constructors.

	/**
	 * Construct a new uninitialized arrow. This constructor is for use only by
	 * object deserialization.
	 */
	public Arrow()
		{
		}

	/**
	 * Construct a new arrow of the given kind and size.
	 *
	 * @param  theKind  Kind of arrow.
	 * @param  theSize  Arrow size.
	 */
	private Arrow
		(int theKind,
		 float theSize)
		{
		myKind = theKind;
		mySize = theSize;
		}

// Exported operations.

	/**
	 * Returns the default arrow size. The arrow size is relative to the stroke
	 * width of the outline to which the arrow is attached.
	 *
	 * @return  Default arrow size.
	 */
	public static float defaultSize()
		{
		return theDefaultSize;
		}

	/**
	 * Set the default arrow size. The arrow size is relative to the stroke
	 * width of the outline to which the arrow is attached. If not set, the
	 * default arrow size is 6.0.
	 *
	 * @param  theSize  Default arrow size.
	 */
	public static void defaultSize
		(float theSize)
		{
		theDefaultSize = theSize;
		}

	/**
	 * Construct a new nonexistent arrow.
	 *
	 * @return  Arrow.
	 */
	public static Arrow none()
		{
		return new Arrow (NONE_KIND, FIXED_SIZE);
		}

	/**
	 * Construct a new solid arrow with the default arrow size.
	 *
	 * @return  Arrow.
	 */
	public static Arrow solid()
		{
		return new Arrow (SOLID_KIND, theDefaultSize);
		}

	/**
	 * Construct a new solid arrow with the given arrow size. The arrow size is
	 * relative to the stroke width of the outline to which the arrow is
	 * attached.
	 *
	 * @param  theSize  Arrow size.
	 *
	 * @return  Arrow.
	 */
	public static Arrow solid
		(float theSize)
		{
		return new Arrow (SOLID_KIND, theSize);
		}

	/**
	 * Construct a new open arrow with the default arrow size.
	 *
	 * @return  Arrow.
	 */
	public static Arrow open()
		{
		return new Arrow (OPEN_KIND, theDefaultSize);
		}

	/**
	 * Construct a new open arrow with the given arrow size. The arrow size is
	 * relative to the stroke width of the outline to which the arrow is
	 * attached.
	 *
	 * @param  theSize  Arrow size.
	 *
	 * @return  Arrow.
	 */
	public static Arrow open
		(float theSize)
		{
		return new Arrow (OPEN_KIND, theSize);
		}

	/**
	 * Write this arrow to the given object output stream.
	 *
	 * @param  out  Object output stream.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void writeExternal
		(ObjectOutput out)
		throws IOException
		{
		out.writeInt (myKind);
		switch (myKind)
			{
			case SOLID_KIND:
			case OPEN_KIND:
				out.writeFloat (mySize);
				break;
			}
		// Note: mySize is written only for SOLID_KIND and OPEN_KIND, for
		// compatibility with previous versions of class Arrow.
		}

	/**
	 * Read this arrow from the given object input stream.
	 *
	 * @param  in  Object input stream.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void readExternal
		(ObjectInput in)
		throws IOException
		{
		myKind = in.readInt();
		if (MIN_KIND > myKind || myKind > MAX_KIND)
			{
			throw new InvalidObjectException
				("Invalid arrow (kind = " + myKind + ")");
			}
		switch (myKind)
			{
			case SOLID_KIND:
			case OPEN_KIND:
				mySize = in.readFloat();
				break;
			default:
				mySize = FIXED_SIZE;
				break;
			}
		// Note: mySize is read only for SOLID_KIND and OPEN_KIND, for
		// compatibility with previous versions of class Arrow.
		}

	/**
	 * Draw this arrow in the given graphics context. It assumes the graphics
	 * context's stroke and paint are already set to the correct values. The
	 * arrow is scaled to match the <TT>width</TT>. The arrow's tip is placed at
	 * the coordinates <TT>(x,y)</TT>. The arrow is rotated so it points in the
	 * direction given by <TT>phi</TT>.
	 *
	 * @param  g2d    2-D graphics context.
	 * @param  width  Stroke width.
	 * @param  x      X coordinate of the arrow's tip.
	 * @param  y      Y coordinate of the arrow's tip.
	 * @param  phi    Angle in which the arrow points (radians).
	 */
	public void draw
		(Graphics2D g2d,
		 float width,
		 double x,
		 double y,
		 double phi)
		{
		float h, w;
		GeneralPath path = null;
		switch (myKind)
			{
			case SOLID_FIXED_KIND:
			case SOLID_KIND:
				h = mySize*GOLDEN_RATIO;
				w = mySize*0.5f;
				path = new GeneralPath();
				path.moveTo (-h, -w);
				path.lineTo (0, 0);
				path.lineTo (-h, w);
				path.closePath();
				break;
			case OPEN_FIXED_KIND:
			case OPEN_KIND:
				h = mySize*GOLDEN_RATIO;
				w = mySize*0.5f;
				path = new GeneralPath();
				path.moveTo (-h, -w);
				path.lineTo (0, 0);
				path.lineTo (-h, w);
				break;
			}

		if (path != null)
			{
			AffineTransform pathTransform = new AffineTransform();
			pathTransform.translate (x, y);
			pathTransform.rotate (phi);
			pathTransform.scale (width, width);
			Shape transformedPath = path.createTransformedShape (pathTransform);
			switch (myKind)
				{
				case SOLID_FIXED_KIND:
				case SOLID_KIND:
					g2d.fill (transformedPath);
					break;
				}
			g2d.draw (transformedPath);
			}
		}

	}
