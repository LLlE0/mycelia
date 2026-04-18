package coordinator

import (
	"time"

	"go.mongodb.org/mongo-driver/bson/primitive"
)

// Participant represents a network participant
type Participant struct {
	ID        primitive.ObjectID     `bson:"_id,omitempty" json:"id"`
	Name      string                 `bson:"name" json:"name"`
	Address   string                 `bson:"address" json:"address"`
	Port      int                    `bson:"port" json:"port"`
	Status    string                 `bson:"status" json:"status"`
	Role      string                 `bson:"role" json:"role"` // PREP, PROC, COORD
	System    map[string]interface{} `bson:"system" json:"system"`
	CreatedAt time.Time              `bson:"created_at" json:"created_at"`
	LastSeen  time.Time              `bson:"last_seen" json:"last_seen"`
}

// Message represents a chat message between participants
type Message struct {
	ID        primitive.ObjectID `bson:"_id,omitempty" json:"id"`
	From      string             `bson:"from" json:"from"`
	To        string             `bson:"to" json:"to"`
	Content   string             `bson:"content" json:"content"`
	Timestamp time.Time          `bson:"timestamp" json:"timestamp"`
}

// TrainingJob represents a federated learning training job
type TrainingJob struct {
	ID              primitive.ObjectID `bson:"_id,omitempty" json:"id"`
	Name            string             `bson:"name" json:"name"`
	Description     string             `bson:"description" json:"description"`
	DatasetURL      string             `bson:"dataset_url" json:"dataset_url"`   // URL to dataset
	DatasetType     string             `bson:"dataset_type" json:"dataset_type"` // csv, json, text, etc.
	DatasetSize     int                `bson:"dataset_size" json:"dataset_size"` // total records in dataset
	ModelType       string             `bson:"model_type" json:"model_type"`     // bert, lstm, cnn, etc.
	ModelName       string             `bson:"model_name" json:"model_name"`     // specific model name
	TotalBatches    int                `bson:"total_batches" json:"total_batches"`
	BatchSize       int                `bson:"batch_size" json:"batch_size"`
	Epochs          int                `bson:"epochs" json:"epochs"`
	LearningRate    float64            `bson:"learning_rate" json:"learning_rate"`
	Status          string             `bson:"status" json:"status"` // pending, preprocessing, training, completed, failed
	CurrentRound    int                `bson:"current_round" json:"current_round"`
	MaxRounds       int                `bson:"max_rounds" json:"max_rounds"`
	Threshold       float64            `bson:"threshold" json:"threshold"` // loss threshold to stop
	CurrentLoss     float64            `bson:"current_loss" json:"current_loss"`
	Progress        float64            `bson:"progress" json:"progress"` // 0-100%
	CreatedAt       time.Time          `bson:"created_at" json:"created_at"`
	UpdatedAt       time.Time          `bson:"updated_at" json:"updated_at"`
	CompletedAt     *time.Time         `bson:"completed_at,omitempty" json:"completed_at,omitempty"`
	ModelWeightsURL string             `bson:"model_weights_url" json:"model_weights_url"` // final model weights
}

// Batch represents a preprocessed data batch stored on PREP node
type Batch struct {
	ID          primitive.ObjectID `bson:"_id,omitempty" json:"id"`
	JobID       string             `bson:"job_id" json:"job_id"`
	BatchNumber int                `bson:"batch_number" json:"batch_number"`
	Status      string             `bson:"status" json:"status"` // pending, ready, in_progress, completed
	RecordCount int                `bson:"record_count" json:"record_count"`
	StoredOn    string             `bson:"stored_on" json:"stored_on"` // node name
	Progress    float64            `bson:"progress" json:"progress"`   // 0-100%
	CreatedAt   time.Time          `bson:"created_at" json:"created_at"`
	UpdatedAt   time.Time          `bson:"updated_at" json:"updated_at"`
}

// ModelUpdate represents weight deltas from a training node
type ModelUpdate struct {
	ID          primitive.ObjectID     `bson:"_id,omitempty" json:"id"`
	JobID       string                 `bson:"job_id" json:"job_id"`
	Round       int                    `bson:"round" json:"round"`
	From        string                 `bson:"from" json:"from"`       // node name
	Status      string                 `bson:"status" json:"status"`   // pending, received, applied
	Weights     map[string]interface{} `bson:"weights" json:"weights"` // weight deltas
	Loss        float64                `bson:"loss" json:"loss"`
	Accuracy    float64                `bson:"accuracy" json:"accuracy"`
	SampleCount int                    `bson:"sample_count" json:"sample_count"`
	Timestamp   time.Time              `bson:"timestamp" json:"timestamp"`
}

// TrainingRound represents a round of federated learning
type TrainingRound struct {
	ID              primitive.ObjectID     `bson:"_id,omitempty" json:"id"`
	JobID           string                 `bson:"job_id" json:"job_id"`
	RoundNumber     int                    `bson:"round_number" json:"round_number"`
	Status          string                 `bson:"status" json:"status"`                 // in_progress, waiting_updates, aggregating, completed
	SelectedNodes   []string               `bson:"selected_nodes" json:"selected_nodes"` // nodes participating
	ReceivedUpdates int                    `bson:"received_updates" json:"received_updates"`
	ExpectedUpdates int                    `bson:"expected_updates" json:"expected_updates"`
	AvgLoss         float64                `bson:"avg_loss" json:"avg_loss"`
	AvgAccuracy     float64                `bson:"avg_accuracy" json:"avg_accuracy"`
	GlobalWeights   map[string]interface{} `bson:"global_weights" json:"global_weights"` // aggregated weights
	CreatedAt       time.Time              `bson:"created_at" json:"created_at"`
	CompletedAt     *time.Time             `bson:"completed_at,omitempty" json:"completed_at,omitempty"`
}

// Task represents a general distributed task (legacy compatibility)
type Task struct {
	ID          primitive.ObjectID     `bson:"_id,omitempty" json:"id"`
	Name        string                 `bson:"name" json:"name"`
	Type        string                 `bson:"type" json:"type"`     // keybert, train, preprocess, etc.
	Status      string                 `bson:"status" json:"status"` // pending, running, completed, failed
	CreatedAt   time.Time              `bson:"created_at" json:"created_at"`
	UpdatedAt   time.Time              `bson:"updated_at" json:"updated_at"`
	StartedAt   *time.Time             `bson:"started_at,omitempty" json:"started_at,omitempty"`
	CompletedAt *time.Time             `bson:"completed_at,omitempty" json:"completed_at,omitempty"`
	InputData   map[string]interface{} `bson:"input_data" json:"input_data"`
	OutputData  map[string]interface{} `bson:"output_data" json:"output_data"`
	AssignedTo  string                 `bson:"assigned_to" json:"assigned_to"`
	Loss        float64                `bson:"loss" json:"loss"`
	Threshold   float64                `bson:"threshold" json:"threshold"`
}

// PendingTask represents a task waiting to be assigned to a node (for polling)
type PendingTask struct {
	ID         primitive.ObjectID     `bson:"_id,omitempty" json:"id"`
	JobID      string                 `bson:"job_id" json:"job_id"`
	JobName    string                 `bson:"job_name" json:"job_name"`
	TaskType   string                 `bson:"task_type" json:"task_type"` // preprocess, train
	Role       string                 `bson:"role" json:"role"`           // PREP, PROC
	Status     string                 `bson:"status" json:"status"`       // pending, assigned, completed
	AssignedTo string                 `bson:"assigned_to" json:"assigned_to"`
	Data       map[string]interface{} `bson:"data" json:"data"`
	CreatedAt  time.Time              `bson:"created_at" json:"created_at"`
	UpdatedAt  time.Time              `bson:"updated_at" json:"updated_at"`
}

// TaskResult represents a result from a worker node
type TaskResult struct {
	ID         primitive.ObjectID     `bson:"_id,omitempty" json:"id"`
	TaskID     string                 `bson:"task_id" json:"task_id"`
	From       string                 `bson:"from" json:"from"`
	Status     string                 `bson:"status" json:"status"`
	ResultData map[string]interface{} `bson:"result_data" json:"result_data"`
	Timestamp  time.Time              `bson:"timestamp" json:"timestamp"`
}

// PreprocessingTask represents a preprocessing task for a job
type PreprocessingTask struct {
	ID            primitive.ObjectID `bson:"_id,omitempty" json:"id"`
	JobID         string             `bson:"job_id" json:"job_id"`
	TaskID        string             `bson:"task_id" json:"task_id"`
	StartOffset   int                `bson:"start_offset" json:"start_offset"`
	EndOffset     int                `bson:"end_offset" json:"end_offset"`
	StartBatchNum int                `bson:"start_batch_num" json:"start_batch_num"`
	Status        string             `bson:"status" json:"status"`
	AssignedTo    string             `bson:"assigned_to" json:"assigned_to"`
	CreatedAt     time.Time          `bson:"created_at" json:"created_at"`
}
