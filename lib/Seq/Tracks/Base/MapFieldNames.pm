# The database stores, for each position, an array of values, either scalars, arrays,
# or hashes
# 1e6 => { [ { 0 => someValue, 1 => otherValue}, {}, someOtheValue, [20,30,40] ]
# This class translates all 'features' names, into integers, which can be used
# to either store a value in the database keyed on that space efficient integer,
# store a value at that integer's index in some array, or translate any
# integer back to its 'feature' name

use 5.10.0;
use strict;
use warnings;

package Seq::Tracks::Base::MapFieldNames;

use Mouse 2;
use List::Util qw/max/;
use Seq::DBManager;

with 'Seq::Role::Message';

################### Required attributes at construction #######################

# The name is required to identify which track's fields we're making dbNames for
# Since to improve performance, we memoize, keyed on track name
# This makes it important, in long-running processes, where many jobs are executed,
# spanning multiple config files, across which track names may not be unique
# to call "initialize" and clear an old config/job's data
has name => (is => 'ro', isa => 'Str', required => 1);

has fieldNamesMap => (is => 'ro', init_arg => undef, lazy => 1, default => sub { {} });
has fieldDbNamesMap => (is => 'ro', init_arg => undef, lazy => 1, default => sub { {} });

################### Private ##########################
#Under which key fields are mapped in the meta database belonging to the
#consuming class' $self->name
#in roles that extend this role, this key's default can be overloaded
state $metaKey = 'fields';

# The db cannot be held in a static variable, because in a long-running/daemon env.
# multiple databases may be called for; the Seq::DBManager package is configured
# before this class is instantiated, on each run.
has _db => (is => 'ro', init_arg => undef, lazy => 1, default => sub {
  return Seq::DBManager->new();
});

sub getFieldDbName {
  #my ($self, $fieldName) = @_;
  
  #$self = $_[0]
  #$fieldName = $_[1]

  if (! exists $_[0]->fieldNamesMap->{$_[0]->name} ) {
    $_[0]->_fetchMetaFields();
  }

  if(! exists $_[0]->fieldNamesMap->{$_[0]->name}{ $_[1] } ) {
    $_[0]->addMetaField( $_[1] );
  }
  
  if(!defined $_[0]->fieldNamesMap->{$_[0]->name}->{$_[1]} ) {
    $_[0]->log('warn', "getFieldDbName failed to find or make a dbName for $_[1]");
    return;
  }

  return $_[0]->fieldNamesMap->{$_[0]->name}->{$_[1]};
}

#this function returns the human readable name
#expected to be used during database reading operations
#like annotation
#@param <Number> $fieldNumber : the database name
sub getFieldName {
  #my ($self, $fieldNumber) = @_;

  #$self = $_[0]
  #$fieldNumber = $_[1]
  if (! exists $_[0]->fieldNamesMap->{ $_[0]->name } ) {
    $_[0]->_fetchMetaFields();
  }

  if(! defined $_[0]->fieldDbNamesMap->{ $_[0]->name }{ $_[1] } ) {
    $_[0]->log('warn', "getFieldName failed to find a name for $_[1]");
    return;
  }

  return $_[0]->fieldDbNamesMap->{ $_[0]->name }{ $_[1] };
}


sub _fetchMetaFields {
  my $self = shift;

  my $dataHref = $self->_db->dbReadMeta($self->name, $metaKey) ;
  
  #if we don't find anything, just store a new hash reference
  #to keep a consistent data type
  if( !$dataHref ) {
    $self->fieldNamesMap->{$self->name} =  {};
    $self->fieldDbNamesMap->{$self->name} = {};
    return;
  }

  $self->fieldNamesMap->{$self->name} = $dataHref;
  #fieldNames map is name => dbName; dbNamesMap is the inverse
  for my $fieldName (keys %$dataHref) {
    $self->fieldDbNamesMap->{$self->name}{ $dataHref->{$fieldName} } = $fieldName;
  }
}

sub addMetaField {
  my $self = shift;
  my $fieldName = shift;

  my @fieldKeys = keys %{ $self->fieldDbNamesMap->{$self->name} };
  
  my $fieldNumber;
  if(!@fieldKeys) {
    $fieldNumber = 0;
  } else {
    #https://ideone.com/eX3dOh
    $fieldNumber = max(@fieldKeys) + 1;
  }
  
  #need a way of checking if the insertion actually worked
  #but that may be difficult with the currrent LMDB_File API
  #I've had very bad performance returning errors from transactions
  #which are exposed in the C api
  #but I may have mistook one issue for another
  #passing 1 to overwrite existing fields
  #since the below mapping ends up relying on our new values
  $self->_db->dbPatchMeta($self->name, $metaKey, {
    $fieldName => $fieldNumber
  }, 1);

  $self->fieldNamesMap->{$self->name}{$fieldName} = $fieldNumber;
  $self->fieldDbNamesMap->{$self->name}{$fieldNumber} = $fieldName;
}

__PACKAGE__->meta->make_immutable;
1;