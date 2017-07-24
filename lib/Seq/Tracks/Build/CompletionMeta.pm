use 5.10.0;
use strict;
use warnings;

package Seq::Tracks::Build::CompletionMeta;
  
  # Keeps track of track build completion
  # TODO: better error handling, not sure how w/ present LMDB API without perf loss
use Mouse 2;
use namespace::autoclean;
use DDP;

with 'Seq::Role::Message';

has name => ( is => 'ro', isa => 'Str', required => 1 );
has db => (is => 'ro', isa => 'Seq::DBManager', required => 1);
has skip_completion_check => (is => 'ro', isa => 'Bool');

############################ Private attributes ########################
# Instance variable holding completion status for this $self->name db
has _completed => (is => 'ro', init_arg => undef, default => sub{ {} });

state $metaKey = 'completed';
###################### Public Methods ######################
sub okToBuild {
  my ($self, $chr) = @_;

  if($self->_isCompleted($chr) ) {
    if(!$self->db->delete && !$self->skip_completion_check) {
      $self->log('debug', "$chr recorded completed for " . $self->name . ". Skipping");
      return 0;
    }
    # Else we're either erasing or re-creating the db; need to erase completion status
    if(!$self->db->dryRun) {
      $self->_eraseCompletionMeta($chr);
    }
  }

  return 1;
}

sub recordCompletion {
  my ($self, $chr) = @_;

  # Note that is $self->delete is set, dbPatchMeta will result in deletion of 
  # the $chr record, ensuring that recordCompletion becomes a deletion operation
  # Except this is more clear, and better log message.
  if($self->db->delete) {
    return $self->log('debug', "Delete set, not recording completion of $chr for ". $self->name);
  }

  if($self->db->dryRun) {
    return $self->log('debug', "dryRun set, not recording completion of $chr for ". $self->name);
  }

  # overwrite any existing entry for $chr
  my $err = $self->db->dbPatchMeta($self->name, $metaKey, { $chr => 1 }, 1 );

  if($err) {
    $self->log('fatal', $err);
    return;
  }

  $self->_completed->{$chr} = 1;

  $self->log('debug', "Recorded completion of $chr (set to 1) for " . $self->name . " db");
};

########################### Private Methods ############################
sub _eraseCompletionMeta {
  my ($self, $chr) = @_;
  
  # Overwrite any existing entry for $chr
  my $err = $self->db->dbPatchMeta($self->name, $metaKey, { $chr => 0 }, 1 );

  if($err) {
    return $self->log('fatal', $err);
  }

  $self->_completed->{$chr} = 0;

  $self->log('debug', "Erased completion of $chr (set to 0) for " . $self->name . " db");
};

sub _isCompleted {
  my ($self, $chr) = @_;

  if(defined $self->_completed->{$chr} ) {
    return $self->_completed->{$chr};
  }

  my $allCompleted = $self->db->dbReadMeta($self->name, $metaKey);
  
  if($allCompleted && defined $allCompleted->{$chr} && $allCompleted->{$chr} == 1) {
    $self->_completed->{$chr} = 1;
  } else {
    $self->_completed->{$chr} = 0;
  }
  
  return $self->_completed->{$chr};
};

__PACKAGE__->meta->make_immutable;
1;
