import edu.rit.draw.*;
import edu.rit.draw.item.*;

public class fig01
	{
	static final double in = 72.0;
	static final double H = in*1;
	static final double W = in*3/2;
	static final double NI_W = in*1/4;
	static final double SW_H = in*1/4;
	static final double SW_W = in*3/4;
	static final double HGAP = in*1/2;
	static final double VGAP = in*1/4;

	static final Fill LIGHT = new ColorFill() .gray (0.95f);
	static final Fill DARK  = new ColorFill() .gray (0.85f);

	static final Outline NET = new SolidOutline() .width (3.0f);

	public static void main
		(String[] args)
		throws Exception
		{
		Text.defaultAlign (Text.Alignment.CENTER);

		Rectangle frontend = new Rectangle() .height (H) .width (W)
			.fill (LIGHT) .add();
		new Text() .text ("Frontend\nNode") .c (frontend.c()) .add();

		Rectangle ni_1 = new Rectangle() .height (H) .width (NI_W)
			.fill (DARK) .e (frontend.w()) .add();
		new Group() .append (new Text() .text ("Ntwk. Intf."))
			.rotationAngle (-Math.PI/2) .c (ni_1.c()) .add();

		Rectangle ni_2 = new Rectangle() .height (H) .width (NI_W)
			.fill (DARK) .w (frontend.e()) .add();
		new Group() .append (new Text() .text ("Ntwk. Intf."))
			.rotationAngle (Math.PI/2) .c (ni_2.c()) .add();

		Rectangle user = new Rectangle() .height (H/2) .width (W/2)
			.fill (LIGHT) .ne (ni_1.w().w(2*HGAP).s(VGAP/2)) .add();
		new Text() .text ("User\nLogin") .c (user.c()) .add();
		new Line() .to (user.e()) .to (ni_1.sw().n(H/4)) .outline (NET)
			.addFirst();

		Rectangle browser = new Rectangle() .height (H/2) .width (W/2)
			.fill (LIGHT) .se (ni_1.w().w(2*HGAP).n(VGAP/2)) .add();
		new Text() .text ("Web\nBrowser") .c (browser.c()) .add();
		new Line() .to (browser.e()) .to (ni_1.nw().s(H/4)) .outline (NET)
			.addFirst();

		new Text() .text ("Internet") .s (browser.ne().e(HGAP)) .add();

		Rectangle swit = new Rectangle() .height (SW_H) .width (SW_W)
			.fill (DARK) .w (ni_2.e().e(HGAP)) .add();
		new Text() .text ("Switch") .c (swit.c()) .add();
		new Line() .to (swit.c()) .to (frontend.c()) .outline (NET)
			.addFirst();

		for (int i = -1; i <= 1; ++ i)
			{
			Rectangle ni_3 = new Rectangle() .height (H) .width (NI_W)
				.fill (DARK) .w (swit.e().x()+HGAP, swit.e().y()+i*(H+VGAP))
				.add();
			new Group() .append (new Text() .text ("Ntwk. Intf."))
				.rotationAngle (-Math.PI/2) .c (ni_3.c()) .add();
			new Line() .to (swit.c()) .to (ni_3.w()) .outline (NET) .addFirst();
			Rectangle backend = new Rectangle() .height (H) .width (W)
				.fill (LIGHT) .w (ni_3.e()) .add();
			new Text() .text ("Backend\nNode") .c (backend.c()) .add();
			}

		new Text() .text ("Backend\nLocal\nNetwork")
			.s (swit.n().x(), browser.n().y()) .add();

		Drawing.write ("fig01.dwg");
		}
	}
