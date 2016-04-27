import edu.rit.draw.*;
import edu.rit.draw.item.*;

public class fig03
	{
	static final double in = 72.0;
	static final double HGAP = in*2;
	static final double VGAP = in*1/2;
	static final double OFF = in*1/32;

	static final Outline CREATE = new DashedOutline();

	public static void main
		(String[] args)
		throws Exception
		{
		Line job = new Line() .to (0, 0) .vby (24.5*VGAP) .add();
		Line tra = new Line() .to (job.n().e(HGAP)) .vby (24.5*VGAP) .add();
		Line lau = new Line() .to (tra.n().e(HGAP)) .vby (24.5*VGAP) .add();
		Line bac = new Line() .to (lau.n().e(HGAP).s(9.5*VGAP)) .vby (13*VGAP)
			.add();
		new Text() .text ("Job") .s (job.n().n(OFF)) .add();
		new Text() .text ("Tracker") .s (tra.n().n(OFF)) .add();
		new Text() .text ("Launcher") .s (lau.n().n(OFF)) .add();
		new Text() .text ("Backend") .s (bac.n().n(OFF)) .add();

		message (lau, tra, false, 1,  "launcherStarted", "1");
		message (lau, tra, true,  2,  "heartbeat", "2");
		message (job, tra, false, 3,  "launchJob", "3");
		message (tra, job, false, 4,  "jobLaunched", "4");
		message (tra, job, false, 5,  "jobStarted", "5");
		message (job, tra, true,  6,  "heartbeat", "6");
		message (job, tra, false, 7,  "launchTaskGroup", "7");
		message (tra, job, false, 8,  "taskLaunching\ntaskFailed", "8");
		message (tra, lau, false, 9,  "launch", "9");
		create  (lau, bac,        10, "(create)", "10");
		message (lau, tra, false, 11, "launchFailed", "11");
		message (tra, job, false, 11, "taskFailed", null);
		message (bac, job, false, 12, "taskLaunched", "12");
		message (job, bac, false, 13, "startTask", "13");
		message (job, bac, true,  14, "heartbeat", "14");
		message (bac, job, false, 15, "writeStandardStream", "15");
		message (bac, job, false, 16, "takeTuple", "16");
		message (job, bac, false, 17, "tupleTaken", "17");
		message (bac, job, false, 18, "writeTuple", "18");
		message (bac, job, false, 19, "taskFinished\ntaskFailed", "19");
		message (job, tra, false, 20, "taskDone", "20");
		message (job, tra, false, 21, "jobDone", "21");
		message (job, bac, false, 22, "stopTask", "22");
		message (job, tra, false, 23, "stopJob", null);
		message (lau, tra, false, 24, "launcherStopped", "23");

		Drawing.write ("fig03.dwg");
		}

	private static void message
		(Line src,
		 Line dst,
		 boolean both,
		 int V,
		 String msg,
		 String num)
		{
		Line l = new Line() .to (src.n().x(), V*VGAP) .hto (dst.n())
			.endArrow (Arrow.SOLID) .add();
		if (both)
			l.startArrow (Arrow.SOLID);
		Text t = new Text() .text (msg) .add();
		if (both)
			t.s (l.c().n(OFF)) .align (Text.Alignment.CENTER);
		else if (src.n().x() > dst.n().x())
			t.se (l.e().n(OFF).w(3*OFF)) .align (Text.Alignment.RIGHT);
		else
			t.sw (l.w().n(OFF).e(3*OFF)) .align (Text.Alignment.LEFT);
		if (num != null)
			new Text() .text (num) .e (-3*OFF, l.w().y()) .add();
		}

	private static void create
		(Line src,
		 Line dst,
		 int V,
		 String msg,
		 String num)
		{
		Line l = new Line() .to (src.n().x(), V*VGAP) .hto (dst.n())
			.endArrow (Arrow.SOLID) .outline (CREATE) .add();
		Text t = new Text() .text (msg) .add();
		if (src.n().x() > dst.n().x())
			t.se (l.e().n(OFF).w(3*OFF)) .align (Text.Alignment.RIGHT);
		else
			t.sw (l.w().n(OFF).e(3*OFF)) .align (Text.Alignment.LEFT);
		if (num != null)
			new Text() .text (num) .e (-3*OFF, l.w().y()) .add();
		}
	}
