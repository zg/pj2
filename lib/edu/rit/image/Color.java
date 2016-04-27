//******************************************************************************
//
// File:    Color.java
// Package: edu.rit.image
// Unit:    Class edu.rit.image.Color
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
 * Class Color provides a color. The color can be specified as red-green-blue
 * (RGB) components or as hue-saturation-brightness (HSB) components.
 *
 * @author  Alan Kaminsky
 * @version 27-Jun-2014
 */
public class Color
	implements Streamable
	{

// Hidden data members.

	int rgb;

// Exported constructors.

	/**
	 * Construct a new color object. The red, green, and blue components are 0,
	 * yielding the color black.
	 */
	public Color()
		{
		}

	/**
	 * Construct a new color object that is a copy of the given color object.
	 *
	 * @param  color  Color object to copy.
	 */
	public Color
		(Color color)
		{
		this.rgb = color.rgb;
		}

// Exported operations.

	/**
	 * Get the packed RGB representation of this color object. The packed RGB
	 * representation is an <TT>int</TT> with bits 31-24 unused, the red
	 * component in bits 23-16, the green component in bits 15-8, and the blue
	 * component in bits 7-0.
	 *
	 * @return  Packed RGB representation.
	 */
	public int rgb()
		{
		return rgb;
		}

	/**
	 * Get the red component of this color object.
	 *
	 * @return  Red component (0 .. 255).
	 */
	public int red()
		{
		return (rgb >> 16) & 255;
		}

	/**
	 * Get the green component of this color object.
	 *
	 * @return  Green component (0 .. 255).
	 */
	public int green()
		{
		return (rgb >> 8) & 255;
		}

	/**
	 * Get the blue component of this color object.
	 *
	 * @return  Blue component (0 .. 255).
	 */
	public int blue()
		{
		return rgb & 255;
		}

	/**
	 * Set this color object from the given packed RGB representation. The
	 * packed RGB representation is an <TT>int</TT> with bits 31-24 unused, the
	 * red component in bits 23-16, the green component in bits 15-8, and the
	 * blue component in bits 7-0.
	 *
	 * @param  rgb  Packed RGB representation.
	 *
	 * @return  This color object.
	 */
	public Color rgb
		(int rgb)
		{
		this.rgb = rgb & 0x00FFFFFF;
		return this;
		}

	/**
	 * Set this color object from the given integer red, green, and blue
	 * components. Each component is a value in the range 0 .. 255. (Only bits
	 * 7-0 are used.) A component value of 0 is fully dark; a component value of
	 * 255 is fully bright.
	 *
	 * @param  red    Red component (0 .. 255).
	 * @param  green  Green component (0 .. 255).
	 * @param  blue   Blue component (0 .. 255).
	 *
	 * @return  This color object.
	 */
	public Color rgb
		(int red,
		 int green,
		 int blue)
		{
		rgb = packRGB (red, green, blue);
		return this;
		}

	/**
	 * Set this color object from the given floating point red, green, and blue
	 * components. Each component is a value in the range 0.0f .. 1.0f. (Values
	 * outside that range are pinned to that range.) A component value of 0.0f
	 * is fully dark; a component value of 1.0f is fully bright.
	 *
	 * @param  red    Red component (0.0f .. 1.0f).
	 * @param  green  Green component (0.0f .. 1.0f).
	 * @param  blue   Blue component (0.0f .. 1.0f).
	 *
	 * @return  This color object.
	 */
	public Color rgb
		(float red,
		 float green,
		 float blue)
		{
		rgb = packRGB
			((int)(red*256.0f),
			 (int)(green*256.0f),
			 (int)(blue*256.0f));
		return this;
		}

	/**
	 * Set this color object from the given floating point hue, saturation, and
	 * brightness components. Each component is a value in the range 0.0f ..
	 * 1.0f. (Values outside that range are pinned to that range.)
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
	 * @param  hue  Hue component (0.0f .. 1.0f).
	 * @param  sat  Saturation component (0.0f .. 1.0f).
	 * @param  bri  Brightness component (0.0f .. 1.0f).
	 *
	 * @return  This color object.
	 */
	public Color hsb
		(float hue,
		 float sat,
		 float bri)
		{
		rgb = packHSB (hue, sat, bri);
		return this;
		}

	/**
	 * Set this color object to the given integer gray value. The gray value is
	 * in the range 0 .. 255. (Only bits 7-0 are used.) A value of 0 is black; a
	 * value of 255 is white.
	 *
	 * @param  gray  Gray value (0 .. 255).
	 *
	 * @return  This color object.
	 */
	public Color gray
		(int gray)
		{
		rgb = packRGB (gray, gray, gray);
		return this;
		}

	/**
	 * Set this color object to the given floating point gray value. The gray
	 * value is in the range 0.0f .. 1.0f. (Values outside that range are pinned
	 * to that range.) A value of 0.0f is black; a value of 1.0f is white.
	 *
	 * @param  gray  Gray value (0.0f .. 1.0f).
	 *
	 * @return  This color object.
	 */
	public Color gray
		(float gray)
		{
		rgb = packRGB (gray, gray, gray);
		return this;
		}

	/**
	 * Write this color object to the given out stream.
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
		out.writeByte ((byte)red());
		out.writeByte ((byte)green());
		out.writeByte ((byte)blue());
		}

	/**
	 * Read this color object from the given in stream.
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
		rgb = packRGB (in.readByte(), in.readByte(), in.readByte());
		}

	/**
	 * Returns a string version of this color object. The string is in
	 * hexadecimal HTML format; for example, <TT>"#123456"</TT>.
	 *
	 * @return  String version.
	 */
	public String toString()
		{
		return String.format ("#%06X", rgb);
		}

// Hidden operations.

	/**
	 * Pack the given RGB components into an <TT>int</TT>.
	 */
	static int packRGB
		(int red,
		 int green,
		 int blue)
		{
		return (lsb (red) << 16) | (lsb (green) << 8) | lsb (blue);
		}

	/**
	 * Returns the least significant byte of the given <TT>int</TT>.
	 */
	private static int lsb
		(int x)
		{
		return x & 255;
		}

	/**
	 * Pack the given RGB components into an <TT>int</TT>.
	 */
	static int packRGB
		(float red,
		 float green,
		 float blue)
		{
		return packRGB
			((int)(red*256.0f),
			 (int)(green*256.0f),
			 (int)(blue*256.0f));
		}

	/**
	 * Pack the given HSB components into an <TT>int</TT>.
	 */
	static int packHSB
		(float hue,
		 float sat,
		 float bri)
		{
		hue = pin (hue);
		sat = pin (sat);
		bri = pin (bri);

		bri *= 256.0f;

		int red, green, blue;

		if (sat == 0.0f)
			{
			red = green = blue = (int)(bri);
			}
		else
			{
			hue = hue * 6.0f;
			int huecase = (int) hue;
			hue = hue - huecase;
			switch (huecase)
				{
				case 0:
				case 6:
					red   = (int)(bri);
					green = (int)(bri*(1.0f - (sat*(1.0f - hue))));
					blue  = (int)(bri*(1.0f - sat));
					// red >= green >= blue
					break;
				case 1:
					red   = (int)(bri*(1.0f - sat*hue));
					green = (int)(bri);
					blue  = (int)(bri*(1.0f - sat));
					// green >= red >= blue
					break;
				case 2:
					red   = (int)(bri*(1.0f - sat));
					green = (int)(bri);
					blue  = (int)(bri*(1.0f - (sat*(1.0f - hue))));
					// green >= blue >= red
					break;
				case 3:
					red   = (int)(bri*(1.0f - sat));
					green = (int)(bri*(1.0f - sat*hue));
					blue  = (int)(bri);
					// blue >= green >= red
					break;
				case 4:
					red   = (int)(bri*(1.0f - (sat*(1.0f - hue))));
					green = (int)(bri*(1.0f - sat));
					blue  = (int)(bri);
					// blue >= red >= green
					break;
				case 5:
					red   = (int)(bri);
					green = (int)(bri*(1.0f - sat));
					blue  = (int)(bri*(1.0f - sat*hue));
					// red >= blue >= green
					break;
				default:
					red = green = blue = (int)(bri);
					// red == green == blue
					break;
				}
			}

		return packRGB (pin (red), pin (green), pin (blue));
		}

	/**
	 * Pin the given <TT>float</TT> to the range 0.0f .. 1.0f.
	 */
	private static float pin
		(float x)
		{
		return Math.max (0.0f, Math.min (x, 1.0f));
		}

	/**
	 * Pin the given <TT>int</TT> to the range 0 .. 255.
	 */
	private static int pin
		(int x)
		{
		return Math.max (0, Math.min (x, 255));
		}

	}
