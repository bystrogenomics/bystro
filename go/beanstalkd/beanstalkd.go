package beanstalkd

import (
	"fmt"
	"log"
	"os"

	"github.com/beanstalkd/go-beanstalk"
	"github.com/bytedance/sonic"
	"gopkg.in/yaml.v3"
)

const PROGRESS_EVENT = "progress"

type ProgressData struct {
	Progress int `json:"progress"`
	Skipped  int `json:"skipped"`
}

type ProgressMessage struct {
	SubmissionID string       `json:"submissionId"`
	Data         ProgressData `json:"data"`
	Event        string       `json:"event"`
}

type MessageSender interface {
	SetProgress(progress int)
	SendMessage()
	Close() error
}

type BeanstalkdMessageSender struct {
	Message    ProgressMessage
	eventTube  *beanstalk.Tube
	connection *beanstalk.Conn
}

type DebugMessageSender struct {
	Message ProgressMessage
}

// Expected beanstalkd format
//
//	addresses:
//	  - <host1>:<port1>
//	tubes:
//	  index:
//	    submission: index
//	    events: index_events
//	  ...
type BeanstalkdConfig struct {
	Addresses []string `yaml:"addresses"`
	Tubes     struct {
		Annotation struct {
			Submission string `yaml:"submission"`
			Events     string `yaml:"events"`
		} `yaml:"annotation"`
		Ancestry struct {
			Submission string `yaml:"submission"`
			Events     string `yaml:"events"`
		} `yaml:"ancestry"`
		Proteomics struct {
			Submission string `yaml:"submission"`
			Events     string `yaml:"events"`
		} `yaml:"proteomics"`
		PRS struct {
			Submission string `yaml:"submission"`
			Events     string `yaml:"events"`
		} `yaml:"prs"`
		SaveFromQuery struct {
			Submission string `yaml:"submission"`
			Events     string `yaml:"events"`
		} `yaml:"save_from_query"`
		Index struct {
			Submission string `yaml:"submission"`
			Events     string `yaml:"events"`
		} `yaml:"index"`
	} `yaml:"tubes"`
}

type BeanstalkdYAML struct {
	Beanstalkd BeanstalkdConfig `yaml:"beanstalkd"`
}

func (b *BeanstalkdMessageSender) SetProgress(progress int) {
	b.Message.Data.Progress = progress
}

func (d *DebugMessageSender) SetProgress(progress int) {
	d.Message.Data.Progress = progress
}

func (b *BeanstalkdMessageSender) Close() error {
	return b.connection.Close()
}

func (d *DebugMessageSender) Close() error {
	return nil
}

func (b *BeanstalkdMessageSender) SendMessage() {
	messageJson, err := sonic.Marshal(b.Message)

	if err != nil {
		log.Printf("failed to marshall progress message due to: [%s]\n", err)
		return
	}

	b.eventTube.Put(messageJson, 0, 0, 0)
}

func (d *DebugMessageSender) SendMessage() {
	fmt.Printf("Indexed %d\n", d.Message.Data.Progress)
}

func createBeanstalkdConfig(beanstalkConfigPath string) (BeanstalkdConfig, error) {
	var bConfig BeanstalkdYAML

	beanstalkConfig, err := os.ReadFile(beanstalkConfigPath)
	if err != nil {
		return BeanstalkdConfig{}, err
	}

	err = yaml.Unmarshal(beanstalkConfig, &bConfig)
	if err != nil {
		return BeanstalkdConfig{}, err
	}

	return bConfig.Beanstalkd, nil
}

func CreateMessageSender(beanstalkConfigPath string, jobSubmissionID string, tube string) (MessageSender, error) {
	message := ProgressMessage{
		SubmissionID: jobSubmissionID,
		Event:        PROGRESS_EVENT,
		Data: ProgressData{
			Progress: 0,
			Skipped:  0,
		},
	}

	if beanstalkConfigPath == "" {
		return &DebugMessageSender{
			Message: message,
		}, nil
	}

	beanstalkdConfig, err := createBeanstalkdConfig(beanstalkConfigPath)
	if err != nil {
		return nil, err
	}

	beanstalkConnection, err := beanstalk.Dial("tcp", beanstalkdConfig.Addresses[0])
	if err != nil {
		return nil, err
	}

	var eventTube *beanstalk.Tube
	switch tube {
	case "annotation":
		eventTube = beanstalk.NewTube(beanstalkConnection, beanstalkdConfig.Tubes.Annotation.Events)
	case "ancestry":
		eventTube = beanstalk.NewTube(beanstalkConnection, beanstalkdConfig.Tubes.Ancestry.Events)
	case "proteomics":
		eventTube = beanstalk.NewTube(beanstalkConnection, beanstalkdConfig.Tubes.Proteomics.Events)
	case "prs":
		eventTube = beanstalk.NewTube(beanstalkConnection, beanstalkdConfig.Tubes.PRS.Events)
	case "saveFromQuery":
		eventTube = beanstalk.NewTube(beanstalkConnection, beanstalkdConfig.Tubes.SaveFromQuery.Events)
	case "index":
		eventTube = beanstalk.NewTube(beanstalkConnection, beanstalkdConfig.Tubes.Index.Events)
	default:
		return nil, fmt.Errorf("unknown event tube: %s", eventTube)
	}

	return &BeanstalkdMessageSender{
		Message:    message,
		eventTube:  eventTube,
		connection: beanstalkConnection,
	}, nil
}
