# ==============================================================================
# S3 Bucket Configuration for Enterprise Asset Storage
# ==============================================================================

# 1. Primary S3 Bucket Definition
resource "aws_s3_bucket" "document_storage" {
  bucket        = "enterprise-document-storage-prod-eu-central-1"
  force_destroy = false # Protects against accidental bucket deletion in production

  tags = {
    Environment = "Production"
    ManagedBy   = "Terraform"
    Project     = "Multimodal Document QA"
  }
}

# 2. Public Access Block (Strict Isolation)
# Prevents any accidental exposure of confidential documents to the public Internet.
resource "aws_s3_bucket_public_access_block" "document_storage_privacy" {
  bucket = aws_s3_bucket.document_storage.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# 3. Server-Side Encryption (SSE-KMS)
# Encrypts all uploaded assets at rest using AWS KMS managed key.
resource "aws_s3_bucket_server_side_encryption_configuration" "document_storage_encryption" {
  bucket = aws_s3_bucket.document_storage.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "aws:kms"
    }
    bucket_key_enabled = true # Reduces KMS request costs by caching keys at the S3 level
  }
}

# 4. Bucket Versioning
# Retains object version history to allow recovery from accidental overrides or deletions.
resource "aws_s3_bucket_versioning" "document_storage_versioning" {
  bucket = aws_s3_bucket.document_storage.id

  versioning_configuration {
    status = "Enabled"
  }
}

# ==============================================================================
# VPC Endpoint Configuration (Private Network Routing)
# ==============================================================================

# S3 Gateway VPC Endpoint ensures S3 traffic stays within the private AWS network (no NAT Gateway costs).
resource "aws_vpc_endpoint" "s3_endpoint" {
  vpc_id            = aws_vpc.main.id # References your main VPC resource
  service_name      = "com.amazonaws.eu-central-1.s3"
  vpc_endpoint_type = "Gateway"

  route_table_ids = [aws_route_table.private_route_table.id]

  tags = {
    Name = "s3-vpc-gateway-endpoint"
  }
}