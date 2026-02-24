"""Update all tables: rename id to idx, add user table, update fields

Revision ID: b1a2f91a1467
Revises: 57c653bcf618
Create Date: 2026-02-09 12:20:11.990804

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'b1a2f91a1467'
down_revision: Union[str, Sequence[str], None] = '57c653bcf618'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    
    # 1. user 테이블 먼저 수정 (post가 user를 참조하므로)
    op.add_column('user', sa.Column('sex', sa.String(length=16), nullable=True))
    op.add_column('user', sa.Column('hand', sa.String(length=16), nullable=True))
    op.alter_column('user', 'password',
               existing_type=sa.TEXT(),
               type_=sa.String(length=20),
               existing_nullable=False)
    
    # 2. 기존 외래키 제약조건 먼저 모두 삭제
    op.drop_constraint('analysis_post_id_fkey', 'analysis', type_='foreignkey')
    op.drop_constraint('file_post_id_fkey', 'file', type_='foreignkey')
    op.drop_constraint('llm_report_post_id_fkey', 'llm_report', type_='foreignkey')
    
    # 3. post 테이블의 기존 primary key 제약조건 삭제 및 idx 컬럼 추가
    op.drop_constraint('post_pkey', 'post', type_='primary')
    op.add_column('post', sa.Column('idx', sa.String(length=36), nullable=False))
    # idx에 기존 id 값 복사
    op.execute('UPDATE post SET idx = id')
    # idx를 새로운 primary key로 설정
    op.create_primary_key('post_pkey', 'post', ['idx'])
    
    # post 테이블의 user_id 외래키 설정
    op.alter_column('post', 'user_id',
               existing_type=sa.VARCHAR(length=36),
               nullable=False)
    op.create_foreign_key('post_user_id_fkey', 'post', 'user', ['user_id'], ['id'], ondelete='CASCADE')
    
    # 4. analysis 테이블 수정
    op.drop_constraint('analysis_pkey', 'analysis', type_='primary')
    op.add_column('analysis', sa.Column('idx', sa.String(length=36), nullable=False))
    # idx에 기존 id 값 복사
    op.execute('UPDATE analysis SET idx = id')
    
    op.add_column('analysis', sa.Column('post_idx', sa.String(length=36), nullable=False))
    # post_idx에 기존 post_id 값 복사
    op.execute('UPDATE analysis SET post_idx = post_id')
    
    op.add_column('analysis', sa.Column('kf1', sa.Integer(), nullable=True))
    op.add_column('analysis', sa.Column('kf2', sa.Integer(), nullable=True))
    op.add_column('analysis', sa.Column('kf3', sa.Integer(), nullable=True))
    op.add_column('analysis', sa.Column('kf1_error', sa.Float(), nullable=True))
    op.add_column('analysis', sa.Column('kf2_error', sa.Float(), nullable=True))
    op.add_column('analysis', sa.Column('kf3_error', sa.Float(), nullable=True))
    op.add_column('analysis', sa.Column('create_date', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True))
    
    # 새로운 primary key 및 외래키 생성
    op.create_primary_key('analysis_pkey', 'analysis', ['idx'])
    op.create_foreign_key('analysis_post_idx_fkey', 'analysis', 'post', ['post_idx'], ['idx'], ondelete='CASCADE')
    
    # 기존 컬럼 삭제
    op.drop_column('analysis', 'angle_json')
    op.drop_column('analysis', 'keypoint_json')
    op.drop_column('analysis', 'created_at')
    op.drop_column('analysis', 'id')
    op.drop_column('analysis', 'post_id')
    
    # 5. file 테이블 수정
    op.drop_constraint('file_pkey', 'file', type_='primary')
    op.add_column('file', sa.Column('idx', sa.String(length=36), nullable=False))
    # idx에 기존 id 값 복사
    op.execute('UPDATE file SET idx = id')
    
    op.add_column('file', sa.Column('post_idx', sa.String(length=36), nullable=False))
    # post_idx에 기존 post_id 값 복사
    op.execute('UPDATE file SET post_idx = post_id')
    
    op.add_column('file', sa.Column('file_name', sa.String(length=255), nullable=True))
    op.add_column('file', sa.Column('file_extension', sa.String(length=10), nullable=True))
    op.add_column('file', sa.Column('file_size', sa.Integer(), nullable=True))
    op.add_column('file', sa.Column('storage_type', sa.String(length=10), nullable=True))
    op.add_column('file', sa.Column('s3_bucket', sa.String(length=100), nullable=True))
    op.add_column('file', sa.Column('s3_key', sa.String(length=500), nullable=True))
    op.add_column('file', sa.Column('create_date', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True))
    
    op.alter_column('file', 'file_path',
               existing_type=sa.TEXT(),
               type_=sa.String(length=500),
               existing_nullable=False)
    
    # 새로운 primary key 및 외래키 생성
    op.create_primary_key('file_pkey', 'file', ['idx'])
    op.create_foreign_key('file_post_idx_fkey', 'file', 'post', ['post_idx'], ['idx'], ondelete='CASCADE')
    
    # 기존 컬럼 삭제
    op.drop_column('file', 'id')
    op.drop_column('file', 'post_id')
    op.drop_column('file', 'created_at')
    
    # 6. llm_report 테이블 수정
    op.drop_constraint('llm_report_pkey', 'llm_report', type_='primary')
    op.add_column('llm_report', sa.Column('idx', sa.String(length=36), nullable=False))
    # idx에 기존 id 값 복사
    op.execute('UPDATE llm_report SET idx = id')
    
    op.add_column('llm_report', sa.Column('post_idx', sa.String(length=36), nullable=False))
    # post_idx에 기존 post_id 값 복사
    op.execute('UPDATE llm_report SET post_idx = post_id')
    
    op.add_column('llm_report', sa.Column('feedback', postgresql.JSON(astext_type=sa.Text()), nullable=True))
    op.add_column('llm_report', sa.Column('create_date', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True))
    
    # 새로운 primary key 및 외래키 생성
    op.create_primary_key('llm_report_pkey', 'llm_report', ['idx'])
    op.create_foreign_key('llm_report_post_idx_fkey', 'llm_report', 'post', ['post_idx'], ['idx'], ondelete='CASCADE')
    
    # 기존 컬럼 삭제
    op.drop_column('llm_report', 'key_points')
    op.drop_column('llm_report', 'summary')
    op.drop_column('llm_report', 'improvement')
    op.drop_column('llm_report', 'created_at')
    op.drop_column('llm_report', 'id')
    op.drop_column('llm_report', 'post_id')
    
    # 7. 마지막으로 post 테이블의 id 컬럼 삭제
    op.drop_column('post', 'id')


def downgrade() -> None:
    """Downgrade schema."""
    
    # 역순으로 복구
    # 1. post 테이블의 id 컬럼 복구
    op.add_column('post', sa.Column('id', sa.VARCHAR(length=36), autoincrement=False, nullable=False))
    op.execute('UPDATE post SET id = idx')
    
    # 2. 모든 외래키 제약조건 먼저 삭제
    op.drop_constraint('llm_report_post_idx_fkey', 'llm_report', type_='foreignkey')
    op.drop_constraint('file_post_idx_fkey', 'file', type_='foreignkey')
    op.drop_constraint('analysis_post_idx_fkey', 'analysis', type_='foreignkey')
    op.drop_constraint('post_user_id_fkey', 'post', type_='foreignkey')
    
    # 3. llm_report 테이블 복구
    op.add_column('llm_report', sa.Column('post_id', sa.VARCHAR(length=36), autoincrement=False, nullable=False))
    op.add_column('llm_report', sa.Column('id', sa.VARCHAR(length=36), autoincrement=False, nullable=False))
    op.execute('UPDATE llm_report SET post_id = post_idx, id = idx')
    
    op.add_column('llm_report', sa.Column('created_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), autoincrement=False, nullable=True))
    op.add_column('llm_report', sa.Column('improvement', postgresql.JSON(astext_type=sa.Text()), autoincrement=False, nullable=True))
    op.add_column('llm_report', sa.Column('summary', sa.TEXT(), autoincrement=False, nullable=True))
    op.add_column('llm_report', sa.Column('key_points', postgresql.JSON(astext_type=sa.Text()), autoincrement=False, nullable=True))
    
    op.drop_constraint('llm_report_pkey', 'llm_report', type_='primary')
    op.create_primary_key('llm_report_pkey', 'llm_report', ['id'])
    
    op.drop_column('llm_report', 'create_date')
    op.drop_column('llm_report', 'feedback')
    op.drop_column('llm_report', 'post_idx')
    op.drop_column('llm_report', 'idx')
    
    # 4. file 테이블 복구
    op.add_column('file', sa.Column('created_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), autoincrement=False, nullable=True))
    op.add_column('file', sa.Column('post_id', sa.VARCHAR(length=36), autoincrement=False, nullable=False))
    op.add_column('file', sa.Column('id', sa.VARCHAR(length=36), autoincrement=False, nullable=False))
    op.execute('UPDATE file SET post_id = post_idx, id = idx')
    
    op.drop_constraint('file_pkey', 'file', type_='primary')
    op.create_primary_key('file_pkey', 'file', ['id'])
    
    op.alter_column('file', 'file_path',
               existing_type=sa.String(length=500),
               type_=sa.TEXT(),
               existing_nullable=False)
    
    op.drop_column('file', 'create_date')
    op.drop_column('file', 's3_key')
    op.drop_column('file', 's3_bucket')
    op.drop_column('file', 'storage_type')
    op.drop_column('file', 'file_size')
    op.drop_column('file', 'file_extension')
    op.drop_column('file', 'file_name')
    op.drop_column('file', 'post_idx')
    op.drop_column('file', 'idx')
    
    # 5. analysis 테이블 복구
    op.add_column('analysis', sa.Column('post_id', sa.VARCHAR(length=36), autoincrement=False, nullable=False))
    op.add_column('analysis', sa.Column('id', sa.VARCHAR(length=36), autoincrement=False, nullable=False))
    op.execute('UPDATE analysis SET post_id = post_idx, id = idx')
    
    op.add_column('analysis', sa.Column('created_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), autoincrement=False, nullable=True))
    op.add_column('analysis', sa.Column('keypoint_json', postgresql.JSONB(astext_type=sa.Text()), autoincrement=False, nullable=True))
    op.add_column('analysis', sa.Column('angle_json', postgresql.JSONB(astext_type=sa.Text()), autoincrement=False, nullable=True))
    
    op.drop_constraint('analysis_pkey', 'analysis', type_='primary')
    op.create_primary_key('analysis_pkey', 'analysis', ['id'])
    
    op.drop_column('analysis', 'create_date')
    op.drop_column('analysis', 'kf3_error')
    op.drop_column('analysis', 'kf2_error')
    op.drop_column('analysis', 'kf1_error')
    op.drop_column('analysis', 'kf3')
    op.drop_column('analysis', 'kf2')
    op.drop_column('analysis', 'kf1')
    op.drop_column('analysis', 'post_idx')
    op.drop_column('analysis', 'idx')
    
    # 6. post 테이블 복구
    op.alter_column('post', 'user_id',
               existing_type=sa.VARCHAR(length=36),
               nullable=True)
    
    op.drop_constraint('post_pkey', 'post', type_='primary')
    op.create_primary_key('post_pkey', 'post', ['id'])
    op.drop_column('post', 'idx')
    
    # 7. 외래키 복구
    op.create_foreign_key('llm_report_post_id_fkey', 'llm_report', 'post', ['post_id'], ['id'], ondelete='CASCADE')
    op.create_foreign_key('file_post_id_fkey', 'file', 'post', ['post_id'], ['id'], ondelete='CASCADE')
    op.create_foreign_key('analysis_post_id_fkey', 'analysis', 'post', ['post_id'], ['id'], ondelete='CASCADE')
    
    # 8. user 테이블 복구
    op.alter_column('user', 'password',
               existing_type=sa.String(length=20),
               type_=sa.TEXT(),
               existing_nullable=False)
    op.drop_column('user', 'hand')
    op.drop_column('user', 'sex')