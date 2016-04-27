//******************************************************************************
//
// File:    PngSetupDialog.java
// Package: edu.rit.swing
// Unit:    Class edu.rit.swing.PngSetupDialog
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

package edu.rit.swing;

import java.awt.Container;
import java.awt.Frame;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;

import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JDialog;
import javax.swing.JLabel;
import javax.swing.JPanel;

import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;

/**
 * Class PngSetupDialog provides a modal dialog for specifying the scale factor
 * and resolution of a PNG file in the {@linkplain View} program.
 *
 * @author  Alan Kaminsky
 * @version 25-Jul-2013
 */
class PngSetupDialog
	extends JDialog
	{

// Hidden data members.

	private static final int GAP = 10;

	private int width;
	private int height;

	private IntegerTextField scaleFactor;
	private IntegerTextField resolution;
	private JLabel pixelDimensionsLabel;
	private JLabel physicalDimensionsLabel;

	private boolean okButtonClicked;

	private DocumentListener docListener = new DocumentListener()
		{
		public void changedUpdate (DocumentEvent e)
			{
			updateDimensions();
			}
		public void insertUpdate (DocumentEvent e)
			{
			updateDimensions();
			}
		public void removeUpdate (DocumentEvent e)
			{
			updateDimensions();
			}
		};

// Exported constructors.

	/**
	 * Construct a new PNG setup dialog.
	 *
	 * @param  owner   Frame in which the dialog is displayed.
	 * @param  width   Image width (pixels).
	 * @param  height  Image height (pixels).
	 * @param  scale   Initial scale factor (percent).
	 * @param  resol   Initial resolution (pixels/inch).
	 */
	public PngSetupDialog
		(Frame owner,
		 int width,
		 int height,
		 int scale,
		 int resol)
		{
		super (owner, "PNG Setup", true);
		this.width = width;
		this.height = height;

		GridBagConstraints c;
		JLabel l;

		// Set up grid bag layout manager.
		Container pane = getContentPane();
		GridBagLayout layout = new GridBagLayout();
		pane.setLayout (layout);

		// Widgets for scale factor.
		l = new JLabel ("Scale factor (percent)");
		c = new GridBagConstraints();
		c.gridx = 0;
		c.gridy = 0;
		c.insets = new Insets (GAP, GAP, GAP, GAP);
		c.anchor = GridBagConstraints.WEST;
		layout.setConstraints (l, c);
		pane.add (l);
		scaleFactor = new IntegerTextField (scale, 5);
		c = new GridBagConstraints();
		c.gridx = 1;
		c.gridy = 0;
		c.insets = new Insets (GAP, 0, GAP, GAP);
		c.anchor = GridBagConstraints.WEST;
		c.weightx = 1.0;
		layout.setConstraints (scaleFactor, c);
		pane.add (scaleFactor);
		scaleFactor.getDocument().addDocumentListener (docListener);

		// Widgets for resolution.
		l = new JLabel ("Resolution (pixels/inch)");
		c = new GridBagConstraints();
		c.gridx = 0;
		c.gridy = 1;
		c.insets = new Insets (0, GAP, GAP, GAP);
		c.anchor = GridBagConstraints.WEST;
		layout.setConstraints (l, c);
		pane.add (l);
		resolution = new IntegerTextField (resol, 5);
		c = new GridBagConstraints();
		c.gridx = 1;
		c.gridy = 1;
		c.insets = new Insets (0, 0, GAP, GAP);
		c.anchor = GridBagConstraints.WEST;
		c.weightx = 1.0;
		layout.setConstraints (resolution, c);
		pane.add (resolution);
		resolution.getDocument().addDocumentListener (docListener);

		// Widgets for pixel dimensions.
		pixelDimensionsLabel = new JLabel (pixelDimensionsString());
		c = new GridBagConstraints();
		c.gridx = 0;
		c.gridy = 2;
		c.gridwidth = GridBagConstraints.REMAINDER;
		c.insets = new Insets (0, GAP, GAP, GAP);
		c.anchor = GridBagConstraints.WEST;
		c.weightx = 1.0;
		c.weighty = 1.0;
		layout.setConstraints (pixelDimensionsLabel, c);
		pane.add (pixelDimensionsLabel);

		// Widgets for physical dimensions.
		physicalDimensionsLabel = new JLabel (physicalDimensionsString());
		c = new GridBagConstraints();
		c.gridx = 0;
		c.gridy = 3;
		c.gridwidth = GridBagConstraints.REMAINDER;
		c.insets = new Insets (0, GAP, GAP, GAP);
		c.anchor = GridBagConstraints.WEST;
		c.weightx = 1.0;
		c.weighty = 1.0;
		layout.setConstraints (physicalDimensionsLabel, c);
		pane.add (physicalDimensionsLabel);

		// "OK" and "Cancel" buttons.
		JPanel buttons = new JPanel();
		buttons.setLayout (new BoxLayout (buttons, BoxLayout.X_AXIS));
		JButton okButton = new JButton ("OK");
		buttons.add (okButton);
		okButton.addActionListener (new ActionListener()
			{
			public void actionPerformed (ActionEvent e)
				{
				doOkay();
				}
			});
		JButton cancelButton = new JButton ("Cancel");
		buttons.add (cancelButton);
		cancelButton.addActionListener (new ActionListener()
			{
			public void actionPerformed (ActionEvent e)
				{
				doCancel();
				}
			});
		c = new GridBagConstraints();
		c.gridx = 0;
		c.gridy = 4;
		c.gridwidth = GridBagConstraints.REMAINDER;
		c.insets = new Insets (0, GAP, GAP, GAP);
		c.anchor = GridBagConstraints.CENTER;
		c.weightx = 1.0;
		c.weighty = 1.0;
		layout.setConstraints (buttons, c);
		pane.add (buttons);
		getRootPane().setDefaultButton (okButton);

		// Set up window closing actions.
		setDefaultCloseOperation (JDialog.DO_NOTHING_ON_CLOSE);
		addWindowListener (new WindowAdapter()
			{
			public void windowActivated (WindowEvent e)
				{
				scaleFactor.setSelectionStart (0);
				scaleFactor.setSelectionEnd (Integer.MAX_VALUE);
				scaleFactor.requestFocusInWindow();
				}
			public void windowClosing (WindowEvent e)
				{
				doCancel();
				}
			});

		pack();
		}

// Exported operations.

	/**
	 * Determine if the "OK" button was clicked.
	 *
	 * @return  True if the "OK" button was clicked, false otherwise.
	 */
	public boolean isOkay()
		{
		return okButtonClicked;
		}

	/**
	 * Get the scale factor.
	 *
	 * @return  Scale factor (percent).
	 */
	public int getScaleFactor()
		{
		return scaleFactor.value();
		}

	/**
	 * Get the resolution.
	 *
	 * @return  Resolution (pixels/inch).
	 */
	public int getResolution()
		{
		return resolution.value();
		}

// Hidden operations.

	/**
	 * Returns the scaled width in pixels.
	 */
	private int scaledWidth()
		{
		return (int)(scaleFactor.value()/100.0*width + 0.5);
		}

	/**
	 * Returns the scaled height in pixels.
	 */
	private int scaledHeight()
		{
		return (int)(scaleFactor.value()/100.0*height + 0.5);
		}

	/**
	 * Returns the pixel dimensions string.
	 */
	private String pixelDimensionsString()
		{
		return String.format ("%d x %d pixels", scaledWidth(), scaledHeight());
		}

	/**
	 * Returns the pixel dimensions string.
	 */
	private String physicalDimensionsString()
		{
		int r = resolution.value();
		double dr = r == 0 ? 1.0 : (double)r;
		return String.format ("%.2f x %.2f inches",
			scaledWidth()/dr, scaledHeight()/dr);
		}

	/**
	 * Processing when the scale factor or resolution is changed.
	 */
	private void updateDimensions()
		{
		pixelDimensionsLabel.setText (pixelDimensionsString());
		physicalDimensionsLabel.setText (physicalDimensionsString());
		}

	/**
	 * Processing when the "Okay" button is clicked.
	 */
	private void doOkay()
		{
		if (scaleFactor.isOkay (1, Integer.MAX_VALUE) &&
				resolution.isOkay (1, Integer.MAX_VALUE))
			{
			okButtonClicked = true;
			setVisible (false);
			}
		}

	/**
	 * Processing when the "Cancel" button is clicked.
	 */
	private void doCancel()
		{
		okButtonClicked = false;
		setVisible (false);
		}

	}
