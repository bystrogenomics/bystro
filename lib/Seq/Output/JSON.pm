package Seq::Output;
use 5.10.0;
use strict;
use warnings;

use Mouse 2;
use Search::Elasticsearch;
# use Search::Elasticsearch::Async;
use Scalar::Util qw/looks_like_number/;

# my $e = Search::Elasticsearch->new({
#   nodes => 'genome.local:9200',
# });

use DDP;

has outputDataFields => (
  is => 'ro',
  isa => 'ArrayRef',
  lazy => 1,
  default => sub { [] },
  writer => 'setOutputDataFieldsWanted',
);

has secondaryDelimiter => (is => 'ro', default => '|');
# ABSTRACT: Knows how to make an output string
# VERSION

#takes an array of <HashRef> data that is what we grabbed from the database
#and whatever else we added to it
#and an array of <ArrayRef> input data, which contains our original input fields
#which we are going to re-use in our output (namely chr, position, type alleles)
sub makeOutputString {
  my ( $self, $outputDataAref) = @_;

  #open(my $fh, '>', $filePath) or $self->log('fatal', "Couldn't open file $filePath for writing");
  # flatten entry hash references and print to file
  my $outStr = '';
  my $count = 1;

  my $secondDelim = $self->secondaryDelimiter;

  my @out;
  for my $href (@$outputDataAref) {
    
    my @singleLineOutput;
    
    PARENT: for my $feature ( @{$self->outputDataFields} ) {
      if(ref $feature) {
        #it's a trackName => {feature1 => value1, ...}
        my ($parent) = %$feature;

        if(!defined $href->{$parent} ) {
          #https://ideone.com/v9ffO7
          push @singleLineOutput, map { 'NA' } @{ $feature->{$parent} };
          next PARENT;
        }

        if(!ref $href->{$parent}) {
          push @singleLineOutput, $href->{$parent};
          next PARENT;
        }

        CHILD: for my $child (@{ $feature->{$parent} } ) {
          if(!defined $href->{$parent}{$child} ) {
            push @singleLineOutput, 'NA';
            next CHILD;
          }

          if(!ref $href->{$parent}{$child} ) {
            push @singleLineOutput, $href->{$parent}{$child};
            next CHILD;
          }

          # Empty array
          if( !@{ $href->{$parent}{$child} } ) {
            push @singleLineOutput, 'NA';
            next PARENT;
          }

          my $accum = '';
          ACCUM: foreach ( @{  $href->{$parent}{$child} } ) {
            if(!defined $_) {
              $accum .= 'NA;';
              next ACCUM;
            }
            # we could have an array of arrays, separate those by commas
            if(ref $_) {
              for my $val (@{$_}) {
                $accum .= defined $val ? "$val;" : 'NA;';
              }
              chop $accum;
              $accum .= $secondDelim;
              next ACCUM;
            }

            $accum .= "$_;";
          }

          chop $accum;
          push @singleLineOutput, $accum;
        }
        next PARENT;
      }

      ### This could be split into separate function, and used 2x;
      ### kept like this in case perf matters

      #say "feature is $feature";
      #p $href->{feature};
      if(!defined $href->{$feature} ) {
        push @singleLineOutput, 'NA';
        next PARENT;
      }

      if(!ref $href->{$feature} ) {
        push @singleLineOutput, $href->{$feature};
        next PARENT;
      }

      if(! @{ $href->{$feature} } ) {
        push @singleLineOutput, 'NA';
        next PARENT;
      }

      # if( @{ $href->{$feature} } == 1 && !ref $href->{$feature}[0] ) {
      #   push @singleLineOutput, defined $href->{$feature}[0] ? $href->{$feature}[0] : 'NA';
      #   next PARENT;
      # }

      #TODO: could break this out into separate function;
      #need to evaluate performance implications

      ACCUM: foreach ( @{ $href->{$feature} } ) {
        if(!defined $_) {
          $href->{$feature} = 'NA';
          next ACCUM;
        }

        # we could have an array of arrays, separate those by commas
        if(ref $_) {
          for my $val (@{$_}) {
            # perl allows this modification of an array value in a loop
            if(!defined $val) {
              $val = 'NA';
            }
          }
          chop $accum;
          $accum .= $secondDelim;
          next ACCUM;
        }

        $accum .= "$_;";
      }

      chop $accum;
      push @singleLineOutput, $accum;
    }

    push @out, \@singleLineOutput;
  }
  
  return $outStr;
}

# TODO: In Go, or other strongly typed languages, type should be controlled
# by the tracks. In Perl it carriers no benefit, except here, so keeping here
# Otherwise, the Perl Elasticsearch client seems to treat strings that look like a number
# as a string
# Oh, and the reason we don't store all numbers as numbers in the db is because
# we save space, because Perl's msgpack library doesn't support single-precision
# floats.
sub indexOutput {
  my ($self, $outputDataAref) = @_;

  # my $bulk = $e->bulk_helper(
  #   index   => 'test_job6', type => 'job',
  # );

  my @out;
  my $count = 1;
  for my $href (@$outputDataAref) {
    my %doc;
    PARENT: for my $feature ( @{$self->outputDataFields} ) {
      if(ref $feature) {
          #it's a trackName => {feature1 => value1, ...}
          my ($parent) = %$feature;

          CHILD: for my $child (@{ $feature->{$parent} } ) {
            my $value;
            if(defined $href->{$parent}{$child} && looks_like_number($href->{$parent}{$child} ) ) {
              $value = 0 + $href->{$parent}{$child};
            }

            if(index($child, ".") > -1) {
              my @parts = split(/\./, $child);
              $doc{$parent}{$parts[0]}{$parts[1]} = $value;
              next CHILD;
            }

            $doc{$parent}{$child} = $value;
          }
          next PARENT;
      }
      
      if(defined $href->{$feature} && looks_like_number($href->{$feature} ) ) {
        $doc{$feature} = 0 + $href->{$feature};
        next PARENT;
      }

      $doc{$feature} = $href->{$feature};
      push @out, \%doc;
    }
  }
  # $bulk->index({
  #     source => \@out,
  #   });
}
__PACKAGE__->meta->make_immutable;
1;