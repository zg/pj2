import edu.rit.draw.*;
import edu.rit.draw.item.*;

public class fig02
	{
	static final double in = 72.0;
	static final double H = in*2;
	static final double W = in*3;
	static final double OH = in*3/8;
	static final double OW = in*5/8;
	static final double UH = in*1/2;
	static final double UW = in*3/4;
	static final double HGAP = in*1;
	static final double VGAP = in*1/4;
	static final double OFF = in*1/16;
	static final double D = in*1;
	static final double M = in*1/8;

	static final Fill LIGHT = new ColorFill() .gray (0.95f);
	static final Fill DARK  = new ColorFill() .gray (0.85f);

	static final Outline CONN = new SolidOutline() .width (3.0f);
	static final Outline CREATE = new DottedOutline() .width (3.0f);

	public static void main
		(String[] args)
		throws Exception
		{
		Text.defaultAlign (Text.Alignment.CENTER);

		Rectangle frontend = new Rectangle() .height (H) .width (W)
			.fill (LIGHT);
		new Text() .text ("Frontend Node") .c (frontend.n().n(VGAP/2)) .add();

		Ellipse tracker = new Ellipse() .diameter (D)
			.nw (frontend.nw().e(M).s(M)) .add();
		new Text() .text ("Tracker") .c (tracker.c()) .add();

		Ellipse pj2 = new Ellipse() .diameter (D) .se (frontend.se().w(M).n(M))
			.add();
		new Text() .text ("pj2") .n (pj2.n().s(OFF)) .add();

		Rectangle job = new Rectangle() .height (OH) .width (OW) .fill (DARK)
			.c (pj2.c().s(OFF)) .add();
		new Text() .text ("Job") .c (job.c()) .add();

		Rectangle[] node = new Rectangle [3];
		Ellipse[] launcher = new Ellipse [3];
		Ellipse[] backend = new Ellipse [3];
		Rectangle[] task = new Rectangle [3];
		for (int i = 0; i <= 2; ++ i)
			{
			node[i] = new Rectangle() .height (H) .width (W) .fill (LIGHT)
				.w (frontend.e().x()+HGAP, frontend.e().y()+(i-1)*(H+VGAP));
			new Text() .text ("Backend Node") .c (node[i].n().n(VGAP/2)) .add();
			launcher[i] = new Ellipse() .diameter (D)
				.nw (node[i].nw().e(M).s(M)) .add();
			new Text() .text ("Launcher") .c (launcher[i].c()) .add();
			backend[i] = new Ellipse() .diameter (D)
				.se (node[i].se().w(M).n(M)) .add();
			new Text() .text ("Backend") .n (backend[i].n().s(2.5*OFF)) .add();
			task[i] = new Rectangle() .height (OH) .width (OW) .fill (DARK)
				.c (backend[i].c().s(OFF)) .add();
			new Text() .text ("Task") .c (task[i].c()) .add();
			}

		Rectangle browser = new Rectangle() .height (UH) .width (UW)
			.fill (LIGHT) .e (frontend.w().x()-HGAP, tracker.w().y()) .add();
		new Text() .text ("Web\nBrowser") .c (browser.c()) .add();
		new Text() .text ("\u2190 status") .s (browser.e().e(HGAP/2).n(OFF)) .add();

		Rectangle user = new Rectangle() .height (UH) .width (UW)
			.fill (LIGHT) .e (frontend.w().x()-HGAP, job.w().y()) .add();
		new Text() .text ("User\nLogin") .c (user.c()) .add();
		new Text() .text ("stdin \u2192") .s (user.e().e(HGAP/2).n(OFF)) .add();
		new Text() .text ("\u2190 stdout\n\u2190 stderr")
			.n (user.e().e(HGAP/2).s(OFF)) .add();

		OutlinedItem.defaultOutline (CONN);
		Line.defaultRound (in*1/4);
		new Line() .to (tracker.c()) .to (pj2.c()) .addFirst();
		new Line() .to (tracker.n()) .vto (launcher[0].w())
			.hto (launcher[0].w()) .addFirst();
		new Line() .to (pj2.n()) .vto (backend[0].w())
			.hto (backend[0].w()) .addFirst();
		new Line() .to (tracker.e()) .to (launcher[1].w()) .addFirst();
		new Line() .to (pj2.e()) .to (backend[1].w()) .addFirst();
		new Line() .to (tracker.s()) .vto (launcher[2].w())
			.hto (launcher[2].w()) .addFirst();
		new Line() .to (pj2.s()) .vto (backend[2].w())
			.hto (backend[2].w()) .addFirst();
		new Line() .to (browser.e()) .to (tracker.w()) .addFirst();
		new Line() .to (user.e()) .to (job.w()) .addFirst();
		OutlinedItem.defaultOutline (CREATE);
		for (int i = 0; i <= 2; ++ i)
			new Line() .to (launcher[i].c()) .to (backend[i].c()) .addFirst();

		frontend.addFirst();
		for (int i = 0; i <= 2; ++ i) node[i].addFirst();

		Drawing.write ("fig02.dwg");
		}
	}
