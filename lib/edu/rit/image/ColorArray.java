//******************************************************************************
//
// File:    ColorArray.java
// Package: edu.rit.image
// Unit:    Class edu.rit.image.ColorArray
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

package edu.rit.image;

import edu.rit.io.InStream;
import edu.rit.io.OutStream;
import edu.rit.io.Streamable;
import java.io.IOException;

/**
 * Class ColorArray provides an array of colors. The color of each element can
 * be specified as red-green-blue (RGB) components or as
 * hue-saturation-brightness (HSB) components.
 * <P>
 * <I>Note:</I> An instance of class ColorArray uses less storage than an array
 * of instances of class {@linkplain Color Color}.
 *
 * @author  Alan Kaminsky
 * @version 28-Jun-2014
 */
public class ColorArray
	implements Streamable
	{

// Hidden data members.

	int[] rgb;

// Exported constructors.

	/**
	 * Construct a new zero-length color array.
	 */
	public ColorArray()
		{
		this (0);
		}

	/**
	 * Construct a new color array of the given length. For each element, the
	 * red, green, and blue components are 0, yielding the color black.
	 *
	 * @param  len  Array length.
	 */
	public ColorArray
		(int len)
		{
		rgb = new int [len];
		}

	/**
	 * Construct a new color array that is a copy of the given color array.
	 *
	 * @param  array  Color array to copy.
	 */
	public ColorArray
		(ColorArray array)
		{
		this.rgb = (int[]) array.rgb.clone();
		}

// Exported operations.

	/**
	 * Get this color array's length.
	 *
	 * @return  Array length.
	 */
	public int length()
		{
		return rgb.length;
		}

	/**
	 * Get the color of the given element of this color array.
	 * <P>
	 * <I>Note:</I> The returned color object is a <I>copy</I> of the color
	 * array element. Changing the returned color object will not affect this
	 * color array.
	 *
	 * @param  i  Array index.
	 *
	 * @return  Color object.
	 */
	public Color color
		(int i)
		{
		return new Color() .rgb (rgb[i]);
		}

	/**
	 * Get the packed RGB representation of the given element of this color
	 * array. The packed RGB representation is an <TT>int</TT> with bits 31-24
	 * unused, the red component in bits 23-16, the green component in bits
	 * 15-8, and the blue component in bits 7-0.
	 *
	 * @param  i  Array index.
	 *
	 * @return  Packed RGB representation.
	 */
	public int rgb
		(int i)
		{
		return rgb[i];
		}

	/**
	 * Get the red component of the given element of this color array.
	 *
	 * @param  i  Array index.
	 *
	 * @return  Red component (0 .. 255).
	 */
	public int red
		(int i)
		{
		return (rgb[i] >> 16) & 255;
		}

	/**
	 * Get the green component of the given element of this color array.
	 *
	 * @param  i  Array index.
	 *
	 * @return  Green component (0 .. 255).
	 */
	public int green
		(int i)
		{
		return (rgb[i] >> 8) & 255;
		}

	/**
	 * Get the blue component of the given element of this color array.
	 *
	 * @param  i  Array index.
	 *
	 * @return  Blue component (0 .. 255).
	 */
	public int blue
		(int i)
		{
		return rgb[i] & 255;
		}

	/**
	 * Set the given element of this color array to the given color object.
	 * <P>
	 * <I>Note:</I> The array element is set to a <I>copy</I> of the color
	 * object. Changing the color object will not affect this color array.
	 *
	 * @param  i      Array index.
	 * @param  color  Color object.
	 *
	 * @return  This color array.
	 */
	public ColorArray color
		(int i,
		 Color color)
		{
		rgb[i] = color.rgb;
		return this;
		}

	/**
	 * Set the given element of this color array from the given packed RGB
	 * representation. The packed RGB representation is an <TT>int</TT> with
	 * bits 31-24 unused, the red component in bits 23-16, the green component
	 * in bits 15-8, and the blue component in bits 7-0.
	 *
	 * @param  i    Array index.
	 * @param  rgb  Packed RGB representation.
	 *
	 * @return  This color array.
	 */
	public ColorArray rgb
		(int i,
		 int rgb)
		{
		this.rgb[i] = rgb & 0x00FFFFFF;
		return this;
		}

	/**
	 * Set the given element of this color array from the given integer red,
	 * green, and blue components. Each component is a value in the range 0 ..
	 * 255. (Only bits 7-0 are used.) A component value of 0 is fully dark; a
	 * component value of 255 is fully bright.
	 *
	 * @param  i      Array index.
	 * @param  red    Red component (0 .. 255).
	 * @param  green  Green component (0 .. 255).
	 * @param  blue   Blue component (0 .. 255).
	 *
	 * @return  This color array.
	 */
	public ColorArray rgb
		(int i,
		 int red,
		 int green,
		 int blue)
		{
		rgb[i] = Color.packRGB (red, green, blue);
		return this;
		}

	/**
	 * Set the given element of this color array from the given floating point
	 * red, green, and blue components. Each component is a value in the range
	 * 0.0f .. 1.0f. (Values outside that range are pinned to that range.) A
	 * component value of 0.0f is fully dark; a component value of 1.0f is fully
	 * bright.
	 *
	 * @param  i      Array index.
	 * @param  red    Red component (0.0f .. 1.0f).
	 * @param  green  Green component (0.0f .. 1.0f).
	 * @param  blue   Blue component (0.0f .. 1.0f).
	 *
	 * @return  This color array.
	 */
	public ColorArray rgb
		(int i,
		 float red,
		 float green,
		 float blue)
		{
		rgb[i] = Color.packRGB
			((int)(red*256.0f),
			 (int)(green*256.0f),
			 (int)(blue*256.0f));
		return this;
		}

	/**
	 * Set the given element of this color array from the given floating point
	 * hue, saturation, and brightness components. Each component is a value in
	 * the range 0.0f .. 1.0f. (Values outside that range are pinned to that
	 * range.)
	 * <P>
	 * The hue component gives the basic color. A hue of 0 = red; 1/6 = yellow;
	 * 2/6 = green; 3/6 = cyan; 4/6 = blue; 5/6 = magenta; 1 = red again.
	 * Intermediate hue values yield intermediate colors.
	 * <P>
	 * The saturation component specifies how gray or colored the color is. A
	 * saturation of 0 yields fully gray; a saturation of 1 yields fully
	 * colored. Intermediate saturation values yield mixtures of gray and
	 * colored.
	 * <P>
	 * The brightness component specifies how dark or light the color is. A
	 * brightness of 0 yields fully dark (black); a brightness of 1 yields fully
	 * light (somewhere between white and colored depending on the saturation).
	 * Intermediate brightness values yield somewhere between a gray shade and a
	 * darkened color (depending on the saturation).
	 *
	 * @param  i    Array index.
	 * @param  hue  Hue component (0.0f .. 1.0f).
	 * @param  sat  Saturation component (0.0f .. 1.0f).
	 * @param  bri  Brightness component (0.0f .. 1.0f).
	 *
	 * @return  This color array.
	 */
	public ColorArray hsb
		(int i,
		 float hue,
		 float sat,
		 float bri)
		{
		rgb[i] = Color.packHSB (hue, sat, bri);
		return this;
		}

	/**
	 * Set the given element of this color array to the given integer gray
	 * value. The gray value is in the range 0 .. 255. (Only bits 7-0 are used.)
	 * A value of 0 is black; a value of 255 is white.
	 *
	 * @param  i     Array index.
	 * @param  gray  Gray value (0 .. 255).
	 *
	 * @return  This color array.
	 */
	public ColorArray gray
		(int i,
		 int gray)
		{
		rgb[i] = Color.packRGB (gray, gray, gray);
		return this;
		}

	/**
	 * Set the given element of this color array to the given floating point
	 * gray value. The gray value is in the range 0.0f .. 1.0f. (Values outside
	 * that range are pinned to that range.) A value of 0.0f is black; a value
	 * of 1.0f is white.
	 *
	 * @param  i     Array index.
	 * @param  gray  Gray value (0.0f .. 1.0f).
	 *
	 * @return  This color array.
	 */
	public ColorArray gray
		(int i,
		 float gray)
		{
		rgb[i] = Color.packRGB (gray, gray, gray);
		return this;
		}

	/**
	 * Copy a portion of the given color array into this color array.
	 *
	 * @param  src     Source color array.
	 * @param  srcoff  First index to read in source color array.
	 * @param  dstoff  First index to write in this color array.
	 * @param  len     Number of elements to copy.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if a nonexistent index in the source
	 *     color array would be read or a nonexistent index in this color array
	 *     would be written.
	 */
	public void copy
		(ColorArray src,
		 int srcoff,
		 int dstoff,
		 int len)
		{
		System.arraycopy (src.rgb, srcoff, this.rgb, dstoff, len);
		}

	/**
	 * Write this color array to the given out stream.
	 *
	 * @param  out  Out stream.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void writeOut
		(OutStream out)
		throws IOException
		{
		out.writeUnsignedInt (rgb.length);
		for (int i = 0; i < rgb.length; ++ i)
			{
			out.writeByte ((byte) red (i));
			out.writeByte ((byte) green (i));
			out.writeByte ((byte) blue (i));
			}
		}

	/**
	 * Read this color array from the given in stream.
	 *
	 * @param  in  In stream.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void readIn
		(InStream in)
		throws IOException
		{
		int len = in.readUnsignedInt();
		rgb = new int [len];
		for (int i = 0; i < len; ++ i)
			rgb[i] = Color.packRGB
				(in.readByte(), in.readByte(), in.readByte());
		}

	}
