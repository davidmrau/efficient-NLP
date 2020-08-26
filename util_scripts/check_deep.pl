#!/usr/bin/perl -w

use strict;

# Check a TREC 2020 Deep Learning track submission for various
# common errors:
#      * extra fields
#      * multiple run tags
#      * missing or extraneous topics
#      * invalid retrieved documents (approximate check)
#      * duplicate retrieved documents in a single topic
#      * too many documents retrieved for a topic
# Messages regarding submission are printed to an error log

# Results input file is in the form
#     topic_num Q0 docid rank sim tag



# Change these variable values to the directory in which the error log should be put
my $errlog_dir = ".";

# If more than MAX_ERRORS errors, then stop processing; something drastically
# wrong with the file.
my $MAX_ERRORS = 25; 
# May return up to MAX_RET visit ids per topic
my $MAX_RET_DOCS = 100;
my $MAX_RET_PASS = 1000;
my @topics = (
3505,     23849,   26703,   42255,   42752,   47210,   48792,   50122,
53233,    64647,   67316,   75198,   85020,   86606,   88495,   91576,
99005,   118440,  119821,  121171,  125659,  132622,  135802,  141630,
144862,  156498,  166046,  169208,  174463,  175920,  177604,  181626,
197312,  206106,  227873,  240158,  245052,  246883,  253749,  256942,
257119,  258062,  273695,  302846,  318362,  324585,  330501,  330975,
332593,  336901,  360721,  384356,  390360,  405163,  425632,  426175,
435548,  436707,  444389,  449367,  452915,  463271,  469589,  482726,
514096,  519025,  537060,  537817,  543273,  545355,  555530,  583468,
586148,  590019,  605127,  610265,  611953,  640502,  653399,  655526,
655914,  660198,  673670,  701453,  703782,  708979,  716113,  730539,
735482,  735922,  768208,  779302,  792635,  794223,  794429,  801118,
804066,  808400,  809525,  814183,  819983,  849550,  850358,  877809,
883915,  911232,  914916,  918162,  938400,  940547,  945835,  978031,
985594,  997622,  999466, 1030303, 1037496, 1043135, 1045109, 1049519,
1051399, 1056416, 1064670, 1065636, 1071750, 1103153, 1103791, 1104501,
1105792, 1105860, 1106928, 1106979, 1107315, 1107440, 1108450, 1108466,
1108473, 1108651, 1108729, 1109699, 1109707, 1109850, 1110678, 1112142,
1113042, 1113256, 1114166, 1114286, 1114993, 1115210, 1116380, 1117817,
1117886, 1118370, 1118426, 1119118, 1119543, 1120588, 1121353, 1121879,
1122138, 1122767, 1122843, 1123657, 1124552, 1125632, 1125755, 1126523,
1126738, 1127004, 1127233, 1127540, 1128456, 1129081, 1130705, 1130734,
1130847, 1131069, 1132044, 1132247, 1132532, 1132842, 1132943, 1132950,
1133485, 1133579, 1134094, 1134207, 1134431, 1134680, 1134939, 1134988,
1135268, 1135283, 1135413, 1135626, 1136043, 1136047, 1136769, 1136962
);

my %valid_ids;
my %numret;                     # number of docs retrieved per topic
my $results_file;    		# input param: file to be checked 
my $errlog;                     # file name of error log
my ($q0warn, $num_errors);      # flags for errors detected
my $line;                       # current input line
my ($topic,$q0,$docno,$rank,$sim,$tag,$treats);
my $line_num;                   # current input line number
my $run_id;
my ($entity, $maxret);
my ($i,$t,$last_i);

my $usage = "Usage: $0 task resultsfile\n";
$#ARGV == 1 || die $usage;
my $task = $ARGV[0];
if ($task ne "docs" && $task ne "passages") {
    die "$0: task must be either 'docs' or 'passages', not '$task'\n";
}
$entity = ($task eq "docs") ? "document" : "passage";
$maxret = ($task eq "docs") ? $MAX_RET_DOCS : $MAX_RET_PASS;
$results_file = $ARGV[1];


open RESULTS, "<$results_file" ||
    die "Unable to open results file $results_file: $!\n";

$last_i = -1;
while ( ($i=index($results_file,"/",$last_i+1)) > -1) {
    $last_i = $i;
}
$errlog = $errlog_dir . "/" . substr($results_file,$last_i+1) . ".errlog";
open ERRLOG, ">$errlog" ||
    die "Cannot open error log for writing\n";

for my $t (@topics) {
    $numret{$t} = 0;
}
$q0warn = 0;
$num_errors = 0;
$line_num = 0;
$run_id = "";

while ($line = <RESULTS>) {
    chomp $line;
    next if ($line =~ /^\s*$/);

    undef $tag;
    my @fields = split " ", $line;
    $line_num++;
	
    if (scalar(@fields) == 6) {
	($topic,$q0,$docno,$rank,$sim,$tag) = @fields;
    } else {
	&error("Too few fields (expecting 6)");
	exit 255;
    }
	
    # make sure runtag is ok
    if (! $run_id) {		# first line --- remember tag 
	$run_id = $tag;
	if ($run_id !~ /^[A-Za-z0-9_-]{1,15}$/) {
	    &error("Run tag `$run_id' is malformed (must be 1-15 alphanumeric characters plus '-' and '_')");
	    next;
	}
    }
    else {		       # otherwise just make sure one tag used
	if ($tag ne $run_id) {
	    &error("Run tag inconsistent (`$tag' and `$run_id')");
	    next;
	}
    }
	
    # get topic number
    if (! exists $numret{$topic}) {
	&error("Unknown test topic ($topic)");
	$topic = 0;
	next;
    }
	
	
    # make sure second field is "Q0"
    if ($q0 ne "Q0" && ! $q0warn) {
	$q0warn = 1;
	&error("Field 2 is `$q0' not `Q0'");
    }
    
    # remove leading 0's from rank (but keep final 0!)
    $rank =~ s/^0*//;
    if (! $rank) {
	$rank = "0";
    }
	
    # make sure rank is an integer (a past group put sim in rank field by accident)
    if ($rank !~ /^[0-9-]+$/) {
	&error("Column 4 (rank) `$rank' must be an integer");
    }
	
    # make sure DOCNO has right format and not duplicated
    if (check_docno($docno)) {
	if (exists $valid_ids{$docno} && $valid_ids{$docno} == $topic){
	    &error("$entity $docno retrieved more than once for topic $topic");
	    next;
	}
	$valid_ids{$docno} = $topic;
    } else {
	&error("Unknown $entity id `$docno'");
	next;
    }
    $numret{$topic}++;

}


# Do global checks:
#   error if some topic has no (or too many) documents retrieved for it
#   warn if too few documents retrieved for a topic
foreach $t (@topics) {
    if ($numret{$t} == 0) {
        &error("No ${entity}s retrieved for topic $t");
    }
    elsif ($numret{$t} > $maxret) {
        &error("Too many ${entity}s ($numret{$t}) retrieved for topic $t");
    }
}


print ERRLOG "Finished processing $results_file\n";
close ERRLOG || die "Close failed for error log $errlog: $!\n";
if ($num_errors) {
    exit 255;
}
exit 0;


# print error message, keeping track of total number of errors
sub error {
    my $msg_string = pop(@_);

    print ERRLOG 
	"$0 of $results_file: Error on line $line_num --- $msg_string\n";

    $num_errors++;
    if ($num_errors > $MAX_ERRORS) {
        print ERRLOG "$0 of $results_file: Quit. Too many errors!\n";
        close ERRLOG ||
	    die "Close failed for error log $errlog: $!\n";
	exit 255;
    }
}


# Check for a valid docid for this type of run
#
sub check_docno {
    my ($docno) = @_;

    return ($docno =~ /^D?[0-9]+$/);
}

