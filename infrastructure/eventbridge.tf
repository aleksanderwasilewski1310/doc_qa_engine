data "aws_caller_identity" "current" {}

data "aws_region" "current" {}

locals {
  ecs_cluster_arn = "arn:aws:ecs:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:cluster/doc_qa_engine"
  sns_topic_arn   = "arn:aws:sns:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:SNSForEvents"
}

resource "aws_iam_role" "eventbridge_to_sns" {
  name = "EventBridgeToSNSExecutionRole"

  assume_role_policy = jsonencode({
    Version = "2008-10-17"
    Statement = [
      {
        Sid    = "AllowEventBridgeAssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "events.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })
}

resource "aws_iam_policy" "eventbridge_sns_publish" {
  name        = "EventBridgeSNSPublishPolicy"
  description = "Allow EventBridge to publish ECS task status notifications to SNSForEvents."

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = "sns:Publish"
        Resource = local.sns_topic_arn
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "attach_sns_publish" {
  role       = aws_iam_role.eventbridge_to_sns.name
  policy_arn = aws_iam_policy.eventbridge_sns_publish.arn
}

resource "aws_cloudwatch_event_rule" "ecs_task_status" {
  name           = "ecs-task-status-notifier"
  description    = "Captures RUNNING and STOPPED state changes for doc_qa_engine cluster tasks."
  event_bus_name = "default"

  event_pattern = jsonencode({
    source      = ["aws.ecs"]
    detail-type = ["ECS Task State Change"]
    detail = {
      clusterArn = [local.ecs_cluster_arn]
      lastStatus = ["RUNNING", "STOPPED"]
    }
  })
}

resource "aws_cloudwatch_event_target" "sns_target" {
  rule      = aws_cloudwatch_event_rule.ecs_task_status.name
  target_id = "SendTaskStatusToSNS"
  arn       = local.sns_topic_arn
  role_arn  = aws_iam_role.eventbridge_to_sns.arn

  depends_on = [aws_iam_role_policy_attachment.attach_sns_publish]

  input_transformer {
    input_paths = {
      taskArn = "$.detail.taskArn"
      status  = "$.detail.lastStatus"
    }

    input_template = "\"ECS task <taskArn> is <status>.\""
  }
}
