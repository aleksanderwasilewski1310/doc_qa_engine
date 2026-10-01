resource "aws_dynamodb_table" "chat_history" {
  name         = "doc_qa_chat_history"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "session_id"
  range_key    = "message_id"

  attribute {
    name = "session_id"
    type = "S"
  }

 attribute {
    name = "message_id"
    type = "S"
  }

  ttl {
    attribute_name = "ttl"
    enabled        = true
  }

  tags = {
    Environment = "poc"
    Project     = "doc_qa_engine"
  }
}
