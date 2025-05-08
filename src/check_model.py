from typing import Type
from mteb.abstasks.TaskMetadata import TaskMetadata
from src.customtasks.msmarcowithqe import MSMARCOWithQE, MSMARCOHardNegatives  # ← your_module 이름을 맞게 바꾸세요

def check_class(cls: Type):
    print(f"Checking class: {cls.__name__}")

    # 인스턴스 생성 가능한지 테스트 (metadata는 class attribute라 생성 안 해도 체크 가능)
    try:
        assert hasattr(cls, 'metadata'), "Missing 'metadata' attribute"
        assert isinstance(cls.metadata, TaskMetadata), "'metadata' is not a TaskMetadata instance"
        print(f"  ✅ metadata found: {cls.metadata.name}")
    except Exception as e:
        print(f"  ❌ metadata error: {e}")

    # ignore_identical_ids 속성 확인
    try:
        assert hasattr(cls, 'ignore_identical_ids'), "Missing 'ignore_identical_ids' attribute"
        print(f"  ✅ ignore_identical_ids: {cls.ignore_identical_ids}")
    except Exception as e:
        print(f"  ❌ ignore_identical_ids error: {e}")

    # 상속 관계 확인
    print(f"  ✅ MRO (Method Resolution Order): {[base.__name__ for base in cls.__mro__]}")

if __name__ == "__main__":
    check_class(MSMARCOWithQE)
    check_class(MSMARCOHardNegatives)
