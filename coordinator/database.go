package coordinator

import (
	"context"
	"time"

	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/bson/primitive"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

type Database struct {
	client                      *mongo.Client
	collection                  *mongo.Collection
	msgCollection               *mongo.Collection
	taskCollection              *mongo.Collection
	pendingTaskCollection       *mongo.Collection // For task polling
	jobCollection               *mongo.Collection
	batchCollection             *mongo.Collection
	updateCollection            *mongo.Collection
	roundCollection             *mongo.Collection
	preprocessingTaskCollection *mongo.Collection
}

func NewDatabase(uri, dbName string) (*Database, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	client, err := mongo.Connect(ctx, options.Client().ApplyURI(uri))
	if err != nil {
		return nil, err
	}

	if err := client.Ping(ctx, nil); err != nil {
		return nil, err
	}

	db := client.Database(dbName)
	return &Database{
		client:                      client,
		collection:                  db.Collection("participants"),
		msgCollection:               db.Collection("messages"),
		taskCollection:              db.Collection("tasks"),
		pendingTaskCollection:       db.Collection("pending_tasks"),
		jobCollection:               db.Collection("training_jobs"),
		batchCollection:             db.Collection("batches"),
		updateCollection:            db.Collection("model_updates"),
		roundCollection:             db.Collection("training_rounds"),
		preprocessingTaskCollection: db.Collection("preprocessing_tasks"),
	}, nil
}

func (d *Database) AddParticipant(p *Participant) error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	p.CreatedAt = time.Now()
	p.LastSeen = time.Now()
	p.Status = "online"

	_, err := d.collection.InsertOne(ctx, p)
	return err
}

func (d *Database) GetParticipants() ([]Participant, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	cursor, err := d.collection.Find(ctx, bson.M{})
	if err != nil {
		return nil, err
	}
	defer cursor.Close(ctx)

	var participants []Participant
	if err := cursor.All(ctx, &participants); err != nil {
		return nil, err
	}
	return participants, nil
}

func (d *Database) GetParticipantByName(name string) (*Participant, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	var p Participant
	err := d.collection.FindOne(ctx, bson.M{"name": name}).Decode(&p)
	if err != nil {
		return nil, err
	}
	return &p, nil
}

func (d *Database) UpdateParticipantStatus(name, status string) error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	_, err := d.collection.UpdateOne(ctx,
		bson.M{"name": name},
		bson.M{"$set": bson.M{"status": status, "last_seen": time.Now()}},
	)
	return err
}

func (d *Database) UpdateParticipantSystem(name string, system map[string]interface{}) error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	_, err := d.collection.UpdateOne(ctx,
		bson.M{"name": name},
		bson.M{"$set": bson.M{"system": system, "last_seen": time.Now()}},
	)
	return err
}

func (d *Database) UpdateParticipant(name string, updates bson.M) error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	updates["last_seen"] = time.Now()
	_, err := d.collection.UpdateOne(ctx,
		bson.M{"name": name},
		bson.M{"$set": updates},
	)
	return err
}

func (d *Database) DeleteParticipant(name string) error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	_, err := d.collection.DeleteOne(ctx, bson.M{"name": name})
	return err
}

// GetParticipantsByRole returns all participants with a specific role
func (d *Database) GetParticipantsByRole(role string) ([]Participant, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	cursor, err := d.collection.Find(ctx, bson.M{"role": role})
	if err != nil {
		return nil, err
	}
	defer cursor.Close(ctx)

	var participants []Participant
	if err := cursor.All(ctx, &participants); err != nil {
		return nil, err
	}
	return participants, nil
}

// GetBatchesByStatus returns batches with a specific status
func (d *Database) GetBatchesByStatus(status string) ([]Batch, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	cursor, err := d.batchCollection.Find(ctx, bson.M{"status": status})
	if err != nil {
		return nil, err
	}
	defer cursor.Close(ctx)

	var batches []Batch
	if err := cursor.All(ctx, &batches); err != nil {
		return nil, err
	}
	return batches, nil
}

func (d *Database) SaveMessage(msg *Message) error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	msg.Timestamp = time.Now()
	_, err := d.msgCollection.InsertOne(ctx, msg)
	return err
}

func (d *Database) GetMessages() ([]Message, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	opts := options.Find().SetSort(bson.D{{Key: "timestamp", Value: -1}}).SetLimit(100)
	cursor, err := d.msgCollection.Find(ctx, bson.M{}, opts)
	if err != nil {
		return nil, err
	}
	defer cursor.Close(ctx)

	var messages []Message
	if err := cursor.All(ctx, &messages); err != nil {
		return nil, err
	}
	return messages, nil
}

func (d *Database) GetParticipantAddress(name string) (string, int, error) {
	p, err := d.GetParticipantByName(name)
	if err != nil {
		return "", 0, err
	}
	return p.Address, p.Port, nil
}

// Legacy task methods
func (d *Database) SaveTask(task *Task) error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	task.CreatedAt = time.Now()
	task.UpdatedAt = time.Now()
	_, err := d.taskCollection.InsertOne(ctx, task)
	return err
}

func (d *Database) GetTasks() ([]Task, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	opts := options.Find().SetSort(bson.D{{Key: "created_at", Value: -1}}).SetLimit(100)
	cursor, err := d.taskCollection.Find(ctx, bson.M{}, opts)
	if err != nil {
		return nil, err
	}
	defer cursor.Close(ctx)

	var tasks []Task
	if err := cursor.All(ctx, &tasks); err != nil {
		return nil, err
	}
	return tasks, nil
}

func (d *Database) GetTaskByID(id string) (*Task, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	var task Task
	objID, err := primitive.ObjectIDFromHex(id)
	if err != nil {
		return nil, err
	}
	err = d.taskCollection.FindOne(ctx, bson.M{"_id": objID}).Decode(&task)
	if err != nil {
		return nil, err
	}
	return &task, nil
}

func (d *Database) UpdateTaskStatus(id, status string, output map[string]interface{}) error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	objID, err := primitive.ObjectIDFromHex(id)
	if err != nil {
		return err
	}

	update := bson.M{
		"$set": bson.M{
			"status":     status,
			"updated_at": time.Now(),
		},
	}

	if status == "completed" || status == "failed" {
		now := time.Now()
		update["$set"].(bson.M)["completed_at"] = now
	}

	if output != nil {
		for k, v := range output {
			update["$set"].(bson.M)[k] = v
		}
	}

	_, err = d.taskCollection.UpdateOne(ctx, bson.M{"_id": objID}, update)
	return err
}

func (d *Database) GetTasksByStatus(status string) ([]Task, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	cursor, err := d.taskCollection.Find(ctx, bson.M{"status": status})
	if err != nil {
		return nil, err
	}
	defer cursor.Close(ctx)

	var tasks []Task
	if err := cursor.All(ctx, &tasks); err != nil {
		return nil, err
	}
	return tasks, nil
}

func (d *Database) DeleteTask(id string) error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	objID, err := primitive.ObjectIDFromHex(id)
	if err != nil {
		return err
	}

	_, err = d.taskCollection.DeleteOne(ctx, bson.M{"_id": objID})
	return err
}

// Training Job methods
func (d *Database) SaveTrainingJob(job *TrainingJob) error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	job.CreatedAt = time.Now()
	job.UpdatedAt = time.Now()

	result, err := d.jobCollection.InsertOne(ctx, job)
	if err != nil {
		return err
	}

	// Set the generated ID back to the job
	job.ID = result.InsertedID.(primitive.ObjectID)
	return nil
}

func (d *Database) GetTrainingJobs() ([]TrainingJob, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	opts := options.Find().SetSort(bson.D{{Key: "created_at", Value: -1}}).SetLimit(50)
	cursor, err := d.jobCollection.Find(ctx, bson.M{}, opts)
	if err != nil {
		return nil, err
	}
	defer cursor.Close(ctx)

	var jobs []TrainingJob
	if err := cursor.All(ctx, &jobs); err != nil {
		return nil, err
	}
	return jobs, nil
}

func (d *Database) GetTrainingJobByID(id string) (*TrainingJob, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	var job TrainingJob
	objID, err := primitive.ObjectIDFromHex(id)
	if err != nil {
		return nil, err
	}
	err = d.jobCollection.FindOne(ctx, bson.M{"_id": objID}).Decode(&job)
	if err != nil {
		return nil, err
	}
	return &job, nil
}

func (d *Database) UpdateTrainingJob(id string, update bson.M) error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	objID, err := primitive.ObjectIDFromHex(id)
	if err != nil {
		return err
	}

	update["updated_at"] = time.Now()
	_, err = d.jobCollection.UpdateOne(ctx, bson.M{"_id": objID}, bson.M{"$set": update})
	return err
}

func (d *Database) GetTrainingJobByStatus(status string) ([]TrainingJob, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	cursor, err := d.jobCollection.Find(ctx, bson.M{"status": status})
	if err != nil {
		return nil, err
	}
	defer cursor.Close(ctx)

	var jobs []TrainingJob
	if err := cursor.All(ctx, &jobs); err != nil {
		return nil, err
	}
	return jobs, nil
}

// Batch methods
func (d *Database) SaveBatch(batch *Batch) error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	batch.CreatedAt = time.Now()
	batch.UpdatedAt = time.Now()
	_, err := d.batchCollection.InsertOne(ctx, batch)
	return err
}

func (d *Database) GetBatchesByJob(jobID string) ([]Batch, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	cursor, err := d.batchCollection.Find(ctx, bson.M{"job_id": jobID})
	if err != nil {
		return nil, err
	}
	defer cursor.Close(ctx)

	var batches []Batch
	if err := cursor.All(ctx, &batches); err != nil {
		return nil, err
	}
	return batches, nil
}

func (d *Database) GetReadyBatches(jobID string) ([]Batch, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	cursor, err := d.batchCollection.Find(ctx, bson.M{"job_id": jobID, "status": "ready"})
	if err != nil {
		return nil, err
	}
	defer cursor.Close(ctx)

	var batches []Batch
	if err := cursor.All(ctx, &batches); err != nil {
		return nil, err
	}
	return batches, nil
}

func (d *Database) UpdateBatch(id string, update bson.M) error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	objID, err := primitive.ObjectIDFromHex(id)
	if err != nil {
		return err
	}

	update["updated_at"] = time.Now()
	_, err = d.batchCollection.UpdateOne(ctx, bson.M{"_id": objID}, bson.M{"$set": update})
	return err
}

func (d *Database) GetBatchStats(jobID string) (total, ready, completed int) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	total64, _ := d.batchCollection.CountDocuments(ctx, bson.M{"job_id": jobID})
	ready64, _ := d.batchCollection.CountDocuments(ctx, bson.M{"job_id": jobID, "status": "ready"})
	completed64, _ := d.batchCollection.CountDocuments(ctx, bson.M{"job_id": jobID, "status": "completed"})
	return int(total64), int(ready64), int(completed64)
}

// Model Update methods
func (d *Database) SaveModelUpdate(update *ModelUpdate) error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	update.Timestamp = time.Now()
	_, err := d.updateCollection.InsertOne(ctx, update)
	return err
}

func (d *Database) GetModelUpdatesByJobAndRound(jobID string, round int) ([]ModelUpdate, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	cursor, err := d.updateCollection.Find(ctx, bson.M{"job_id": jobID, "round": round})
	if err != nil {
		return nil, err
	}
	defer cursor.Close(ctx)

	var updates []ModelUpdate
	if err := cursor.All(ctx, &updates); err != nil {
		return nil, err
	}
	return updates, nil
}

// Training Round methods
func (d *Database) SaveTrainingRound(round *TrainingRound) error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	round.CreatedAt = time.Now()
	_, err := d.roundCollection.InsertOne(ctx, round)
	return err
}

// Preprocessing Task methods
func (d *Database) SavePreprocessingTask(task *PreprocessingTask) error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	task.CreatedAt = time.Now()
	_, err := d.preprocessingTaskCollection.InsertOne(ctx, task)
	return err
}

func (d *Database) UpdatePreprocessingTask(id string, update bson.M) error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	objID, err := primitive.ObjectIDFromHex(id)
	if err != nil {
		return err
	}

	_, err = d.preprocessingTaskCollection.UpdateOne(ctx, bson.M{"_id": objID}, bson.M{"$set": update})
	return err
}

func (d *Database) UpdatePreprocessingTaskByTaskID(taskID string, update bson.M) error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	_, err := d.preprocessingTaskCollection.UpdateOne(ctx, bson.M{"task_id": taskID}, bson.M{"$set": update})
	return err
}

func (d *Database) GetPreprocessingTaskByID(id string) (*PreprocessingTask, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	var task PreprocessingTask
	objID, err := primitive.ObjectIDFromHex(id)
	if err != nil {
		return nil, err
	}
	err = d.preprocessingTaskCollection.FindOne(ctx, bson.M{"_id": objID}).Decode(&task)
	if err != nil {
		return nil, err
	}
	return &task, nil
}

func (d *Database) GetPreprocessingTasksByJob(jobID string) ([]PreprocessingTask, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	cursor, err := d.preprocessingTaskCollection.Find(ctx, bson.M{"job_id": jobID})
	if err != nil {
		return nil, err
	}
	defer cursor.Close(ctx)

	var tasks []PreprocessingTask
	if err := cursor.All(ctx, &tasks); err != nil {
		return nil, err
	}
	return tasks, nil
}

func (d *Database) GetPreprocessingTasksByStatus(status string) ([]PreprocessingTask, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	cursor, err := d.preprocessingTaskCollection.Find(ctx, bson.M{"status": status})
	if err != nil {
		return nil, err
	}
	defer cursor.Close(ctx)

	var tasks []PreprocessingTask
	if err := cursor.All(ctx, &tasks); err != nil {
		return nil, err
	}
	return tasks, nil
}

func (d *Database) GetTrainingRound(jobID string, roundNum int) (*TrainingRound, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	var round TrainingRound
	err := d.roundCollection.FindOne(ctx, bson.M{"job_id": jobID, "round_number": roundNum}).Decode(&round)
	if err != nil {
		return nil, err
	}
	return &round, nil
}

func (d *Database) UpdateTrainingRound(id string, update bson.M) error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	objID, err := primitive.ObjectIDFromHex(id)
	if err != nil {
		return err
	}

	_, err = d.roundCollection.UpdateOne(ctx, bson.M{"_id": objID}, bson.M{"$set": update})
	return err
}

// PendingTask methods - for task polling
func (d *Database) SavePendingTask(task *PendingTask) error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	task.CreatedAt = time.Now()
	task.UpdatedAt = time.Now()
	_, err := d.pendingTaskCollection.InsertOne(ctx, task)
	return err
}

func (d *Database) GetPendingTasksByRole(role string) ([]PendingTask, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	cursor, err := d.pendingTaskCollection.Find(ctx, bson.M{"role": role, "status": "pending"})
	if err != nil {
		return nil, err
	}
	defer cursor.Close(ctx)

	var tasks []PendingTask
	if err := cursor.All(ctx, &tasks); err != nil {
		return nil, err
	}
	return tasks, nil
}

func (d *Database) GetPendingTaskByID(id string) (*PendingTask, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	objID, err := primitive.ObjectIDFromHex(id)
	if err != nil {
		return nil, err
	}

	var task PendingTask
	err = d.pendingTaskCollection.FindOne(ctx, bson.M{"_id": objID}).Decode(&task)
	if err != nil {
		return nil, err
	}
	return &task, nil
}

func (d *Database) UpdatePendingTask(id string, update bson.M) error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	objID, err := primitive.ObjectIDFromHex(id)
	if err != nil {
		return err
	}

	update["updated_at"] = time.Now()
	_, err = d.pendingTaskCollection.UpdateOne(ctx, bson.M{"_id": objID}, bson.M{"$set": update})
	return err
}

func (d *Database) DeletePendingTask(id string) error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	objID, err := primitive.ObjectIDFromHex(id)
	if err != nil {
		return err
	}

	_, err = d.pendingTaskCollection.DeleteOne(ctx, bson.M{"_id": objID})
	return err
}

func (d *Database) Close(ctx context.Context) error {
	return d.client.Disconnect(ctx)
}
