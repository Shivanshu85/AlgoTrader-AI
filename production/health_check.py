"""
Health Check Script - Verify all system components are operational
"""

import sys
import json
from datetime import datetime
from typing import Dict, Tuple

import os
from pathlib import Path

# Configure path to import production modules
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_environment_variables() -> Tuple[bool, Dict]:
    """Check required environment variables"""
    required_vars = [
        'POSTGRES_DB',
        'POSTGRES_USER',
        'POSTGRES_HOST',
    ]
    
    missing = [var for var in required_vars if not os.getenv(var)]
    
    return len(missing) == 0, {
        'status': 'healthy' if not missing else 'unhealthy',
        'missing': missing
    }


def check_postgres() -> Tuple[bool, Dict]:
    """Check PostgreSQL connectivity"""
    try:
        import psycopg2
        
        conn = psycopg2.connect(
            dbname=os.getenv('POSTGRES_DB', 'stock_prediction'),
            user=os.getenv('POSTGRES_USER', 'postgres'),
            password=os.getenv('POSTGRES_PASSWORD', 'postgres'),
            host=os.getenv('POSTGRES_HOST', 'localhost'),
            port=os.getenv('POSTGRES_PORT', '5432'),
            connect_timeout=5
        )
        conn.close()
        return True, {'status': 'healthy', 'message': 'Connected to PostgreSQL'}
    except Exception as e:
        return False, {'status': 'unhealthy', 'error': str(e)}


def check_redis() -> Tuple[bool, Dict]:
    """Check Redis connectivity"""
    try:
        import redis
        
        r = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', '6379')),
            password=os.getenv('REDIS_PASSWORD', None),
            db=0,
            socket_connect_timeout=5,
            socket_keepalive=True
        )
        r.ping()
        return True, {'status': 'healthy', 'message': 'Connected to Redis'}
    except Exception as e:
        return False, {'status': 'unhealthy', 'error': str(e)}


def check_python_packages() -> Tuple[bool, Dict]:
    """Check critical Python packages"""
    packages = {
        'torch': 'PyTorch',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'sklearn': 'Scikit-learn',
        'sqlalchemy': 'SQLAlchemy',
    }
    
    missing = []
    for package, name in packages.items():
        try:
            __import__(package)
        except ImportError:
            missing.append(name)
    
    return len(missing) == 0, {
        'status': 'healthy' if not missing else 'unhealthy',
        'missing_packages': missing,
        'installed': len(packages) - len(missing)
    }


def check_directories() -> Tuple[bool, Dict]:
    """Check required directories exist"""
    required_dirs = [
        'data/raw',
        'data/processed',
        'data/features',
        'logs',
        'models',
    ]
    
    missing = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing.append(dir_path)
    
    return len(missing) == 0, {
        'status': 'healthy' if not missing else 'unhealthy',
        'missing_directories': missing,
        'present': len(required_dirs) - len(missing)
    }


def run_health_checks() -> Dict:
    """Run all health checks"""
    checks = {
        'environment_variables': check_environment_variables,
        'postgres': check_postgres,
        'redis': check_redis,
        'python_packages': check_python_packages,
        'directories': check_directories,
    }
    
    results = {
        'timestamp': datetime.utcnow().isoformat(),
        'checks': {},
        'overall_status': 'healthy',
    }
    
    for check_name, check_func in checks.items():
        try:
            passed, details = check_func()
            results['checks'][check_name] = {
                'passed': passed,
                'details': details
            }
            if not passed:
                results['overall_status'] = 'unhealthy'
        except Exception as e:
            results['checks'][check_name] = {
                'passed': False,
                'error': str(e)
            }
            results['overall_status'] = 'unhealthy'
    
    return results


def print_results(results: Dict) -> None:
    """Pretty print health check results"""
    print("\n" + "=" * 80)
    print("SYSTEM HEALTH CHECK")
    print("=" * 80)
    print(f"Timestamp: {results['timestamp']}")
    print(f"Overall Status: {results['overall_status'].upper()}")
    print("-" * 80)
    
    for check_name, check_result in results['checks'].items():
        status = "✅ PASS" if check_result['passed'] else "❌ FAIL"
        print(f"\n{status} - {check_name}")
        
        details = check_result.get('details', {})
        for key, value in details.items():
            if isinstance(value, list) and value:
                print(f"    {key}: {', '.join(map(str, value))}")
            elif isinstance(value, dict):
                for k, v in value.items():
                    print(f"    {k}: {v}")
            elif value:
                print(f"    {key}: {value}")
    
    print("\n" + "=" * 80 + "\n")
    
    return results['overall_status'] == 'healthy'


if __name__ == '__main__':
    results = run_health_checks()
    healthy = print_results(results)
    
    # Exit with appropriate code
    sys.exit(0 if healthy else 1)
