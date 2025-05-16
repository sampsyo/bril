
import com.squareup.moshi.*
import com.squareup.moshi.kotlin.reflect.KotlinJsonAdapterFactory
import java.util.TreeSet

data class BriloopProgram(
    val functions: List<BriloopFunction>,
)

data class BriloopFunction(
    val name: String,
    val args: List<BriloopArg>?,
    val type: BriloopType?,
    val instrs: List<BriloopInstr>,
)

data class BriloopArg(
    val name: String,
    val type: BriloopType?,
)

sealed interface BriloopInstr

data class BriloopOp(
    val op: String,
    val args: List<String>?,
    val value: BriloopValue?,
    val dest: String?,
    val type: BriloopType?,
    val funcs: List<String>?,
) : BriloopInstr

sealed interface BriloopStmt : BriloopInstr

data class BriloopIfStmt(
    val arg: String,
    val tru: List<BriloopInstr>,
) : BriloopStmt

data class BriloopIfThenStmt(
    val arg: String,
    val tru: List<BriloopInstr>,
    val fals: List<BriloopInstr>,
) : BriloopStmt

data class BriloopContinueStmt(
    val value: BriloopValue?,
) : BriloopStmt

data class BriloopBreakStmt(
    val value: BriloopValue?,
) : BriloopStmt

data class BriloopWhileStmt(
    val arg: String,
    val body: List<BriloopInstr>,
) : BriloopStmt

data class BriloopBlockStmt(
    val body: List<BriloopInstr>,
) : BriloopStmt

data class BriloopInstrJson(
    val op: String,
    val dest: String?,
    val type: BriloopType?,
    val value: BriloopValue?,
    val args: List<String>?,
    val funcs: List<String>?,
    val labels: List<String>?,
    val children: List<List<BriloopInstrJson>>?,
)

class BriloopInstrAdapter {

    @FromJson
    fun fromJson(
        json: BriloopInstrJson,
    ) : BriloopInstr? {
        if (json.labels.isNullOrEmpty().not()) {
            throw IllegalArgumentException("Cannot translate $json as it uses labels")
        }

        return when(json.op) {
            "block" -> BriloopBlockStmt(
                body = json.children!!.first().mapNotNull {
                    val briloopinstr: BriloopInstr? = fromJson(it)
                    briloopinstr
                },
            )
            "while" -> BriloopWhileStmt(
                arg = json.args!!.first(),
                body = json.children!!.first().mapNotNull { 
                    val briloopinstr: BriloopInstr? = fromJson(it)
                    briloopinstr
                },
            )
            "if" -> BriloopIfThenStmt(
                arg = json.args!!.first(),
                tru = json.children!!.first().mapNotNull { fromJson(it) },
                fals = json.children!!.getOrNull(1)?.mapNotNull { 
                    val briloopinstr: BriloopInstr? = fromJson(it)
                    briloopinstr
                }.orEmpty(),
            )
            "continue" -> BriloopContinueStmt(
                value = json.value,
            )
            "break" -> BriloopBreakStmt(
                value = json.value,
            )
            else -> BriloopOp(
                op = json.op,
                dest = json.dest,
                type = json.type,
                value = json.value,
                args = json.args.orEmpty(),
                funcs = json.funcs.orEmpty(),
            )
        }
    }

    @ToJson
    fun toJson(
        instr: BriloopInstr?
    ): BriloopInstrJson? {
        return when (instr) {
            is BriloopOp -> BriloopInstrJson(
                op = instr.op,
                dest = instr.dest,
                type = instr.type,
                value = instr.value,
                args = instr.args,
                funcs = instr.funcs,
                labels = null,
                children = null
            )
            is BriloopBlockStmt -> BriloopInstrJson(
                op = "block",
                dest = null,
                type = null,
                value = null,
                args =  null,
                funcs = null,
                labels = null,
                children = listOf(instr.body.mapNotNull { toJson(it) })
            )
            is BriloopWhileStmt -> BriloopInstrJson(
                op = "while",
                dest = null,
                type = null,
                value = null,
                args = listOf(instr.arg),
                funcs = null,
                labels = null,
                children = listOf(instr.body.mapNotNull { toJson(it) })
            )
            is BriloopIfStmt -> BriloopInstrJson(
                op = "if",
                dest = null,
                type = null,
                value = null,
                args = listOf(instr.arg),
                funcs = null,
                labels = null,
                children = listOf(
                    instr.tru.mapNotNull { toJson(it) }
                ),
            )
            is BriloopIfThenStmt -> BriloopInstrJson(
                op = "if",
                dest = null,
                type = null,
                value = null,
                args = listOf(instr.arg),
                funcs = null,
                labels = null,
                children = listOf(
                    instr.tru.mapNotNull { toJson(it) },
                    instr.fals.mapNotNull { toJson(it) }
                ),
            )
            is BriloopContinueStmt -> BriloopInstrJson(
                op = "continue",
                dest = null,
                type = null, 
                value = instr.value,
                args = null,
                funcs = null,
                labels = null,
                children = null,
            )
            is BriloopBreakStmt -> BriloopInstrJson(
                op = "break",
                dest = null,
                type = null,
                value = instr.value,
                args = null,
                funcs = null,
                labels = null,
                children = null,
            )
            else -> null
        }
    }
}

fun BrilArg.toBriloop(): BriloopArg {
    return BriloopArg(
        name = this.name,
        type = this.type.toBriloop()
    )
}

fun BrilInstr.toBriloop(): BriloopInstr? {
    val brilOp = this as? BrilOp
    if (brilOp == null) {
        return null
    }

    val adapter = BrilOpAdapter()
    val briljson = adapter.toJson(brilOp)
    return BriloopOp(
        op = briljson.op,
        args = briljson.args,
        funcs = briljson.funcs,
        value = briljson.value.toBriloop(),
        dest = briljson.dest,
        type = briljson.type.toBriloop(),
    )
}

fun BrilType?.toBriloop(): BriloopType? {
    return when (this) {
        is BrilPrimitiveType -> BriloopPrimitiveType(this.name)
        is BrilParameterizedType -> BriloopParameterizedType(this.map)
        else -> null
    }
}

fun BrilPrimitiveValueType?.toBriloop(): BriloopValue? {
    return when (this) {
        is BrilPrimitiveValueBool -> BriloopValueBoolean(this.value)
        is BrilPrimitiveValueInt -> BriloopValueInt(this.value)
        is BrilPrimitiveValueDouble -> BriloopValueDouble(this.value)
        else -> null
    }
}
