// TODO: Use enums where possible
declare type submissionSchema = {
  _id: string;
  state: string;
  attempts: number;
  log: {
    progress: number;
    skipped: number;
    messages: string[];
    exceptions: string[];
  };
  queueID: string;
  queueStats: {
    // Provide some undefined values
    ttr: number;
    age: number;
  };
  // A date
  date: Date;
  type: string;
  addedFileNames: string[];
};

declare type jobSchema = {
  _id: string; //must be empty OR filled with a unique value; if set to value, will be used server-side
  assembly: string;
  email: string;
  // TODO: enumerate config
  config: {};
  options: {
    index: boolean;
  };
  inputFileName: string;
  // A job may be created form a query rather than an inputFilePath
  inputQuery: string;
  //A file prefix for the tarball that contains the file
  //This is what the name the job
  outputBaseFileName: string;
  name: string;
  results: {};
  submission: submissionSchema;
  search: {
    activeSubmission: submissionSchema;
    archivedSubmissions: submissionSchema[];
    // The fields indexed for this job
    fieldNames: string[];
    indexName: string;
    indexType: string;
    savedResults: jobSchema[];
    queries: [
      {
        queryType: string;
        queryValue: string;
      }
    ];
  };
  type: string;
  _creator: string;
  expireDate: Date;
};

// core, required data to submit a job, track progress, view results
// TODO: get schema from server

//TODO: make the schema properties immutable using Object.defineProperties
export const newSubmission = (creatorID: string) => {
  const sInst: Partial<jobSchema> = {};

  if (creatorID) {
    sInst._creator = creatorID;
  }

  return sInst;
};
